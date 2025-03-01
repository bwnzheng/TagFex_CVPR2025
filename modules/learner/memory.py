from abc import abstractmethod
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from .base import RehearsalLearner

class HerdingIndicesLearner(RehearsalLearner):
    """
    Use herding to manage memory. Memory is stored as indices from the training set and in $self.memory_samples.
    It saves memory usage on the machine for memory containing only raw samples.
    """
    def __init__(self, data_maganger, configs: dict, device, distributed=None) -> None:
        super().__init__(data_maganger, configs, device, distributed)
        self.class_means = [] # (n_classes, feat_dims)
    
    def check_memory(self, memory_indices) -> None:
        assert len(memory_indices) <= self.memory_configs['memory_size'], 'Memory Limit Exceeded.'

    def get_memory(self) -> np.ndarray:
        memory_indices = np.array(self.memory_samples, dtype=np.int32).flatten()
        self.check_memory(memory_indices)
        return memory_indices
    
    @abstractmethod
    def extract_herding_features(self, dataset: Dataset) -> torch.Tensor:
        """
        extract normailzed features for herding selection.
        """
        pass

    def distributed_split_classes(self, global_st, global_ed):
        rank = self.distributed['rank']
        world_size = self.distributed['world_size']

        all_local_classes = np.array_split(np.arange(global_st, global_ed), world_size)
        all_num_classes = [len(classes) for classes in all_local_classes]
        local_classes = all_local_classes[rank]
        return local_classes, all_num_classes
    
    def distributed_gather_classes(self, tensor: torch.Tensor, all_num_classes: List[int]) -> torch.Tensor:
        feature_dim = tensor.shape[-1]
        gathered_tensor_list = [torch.zeros((num_classes, feature_dim), dtype=tensor.dtype, device=self.device) for num_classes in all_num_classes]

        torch.distributed.all_gather(gathered_tensor_list, tensor)

        gathered_tensor = torch.cat(gathered_tensor_list)
        return gathered_tensor

    @torch.no_grad()
    def reduce_memory(self) -> None:
        if len(self.memory_samples) == 0 or self.memory_configs.get('fixed_size'):
            return

        previous_memsize_per_class = len(self.memory_samples[0])
        self.memory_samples = np.array(self.memory_samples)[:, :self.num_exemplars_per_class].tolist()

        if self.distributed is not None:
            rank = self.distributed['rank']
            local_classes, all_num_classes = self.distributed_split_classes(0, len(self.memory_samples))
            # self.print_logger.debug(f'rank{rank} {local_classes}')
        else:
            rank = 0
            local_classes = np.arange(len(self.memory_samples))
        
        new_class_means = []
        prog_bar_desc = f"Task {self.state['cur_task']}/{self.state['num_tasks']} Rank{rank} rdcmem [{local_classes[0]}~{local_classes[-1]}][{previous_memsize_per_class}->{self.num_exemplars_per_class}]"
        prog_bar = tqdm(local_classes, desc=prog_bar_desc, position=rank)
        for class_id in prog_bar:
            # self.print_logger.debug(f'rank{rank} {class_id} in {local_classes}')
            class_indices = self.memory_samples[class_id]
            class_dataset = self.data_manager.get_dataset_by_indices(class_indices, split='train', mode='test')
            
            class_features = self.extract_herding_features(class_dataset)
            class_mean = class_features.mean(0)
            class_mean /= class_mean.norm()

            new_class_means.append(class_mean)
        prog_bar.close()
        
        if self.distributed is not None:
            gathered_class_means = self.distributed_gather_classes(torch.stack(new_class_means), all_num_classes)
            new_class_means = [m for m in gathered_class_means]
        
        self.class_means = new_class_means

    @torch.no_grad()
    def update_memory(self) -> None:
        num_classes = self.state['sofar_num_classes']
        cur_task_num_classes = self.state['cur_task_num_classes']
        num_seen_classes = num_classes - cur_task_num_classes

        if self.distributed is not None:
            rank = self.distributed['rank']
            local_classes, all_num_classes = self.distributed_split_classes(num_seen_classes, num_classes)
            # self.print_logger.debug(f'rank{rank} {local_classes}')
        else:
            rank = 0
            local_classes = np.arange(num_seen_classes, num_classes)
        
        selected_indices = []
        class_means = []
        prog_bar_desc = f"Task {self.state['cur_task']}/{self.state['num_tasks']} Rank{rank} updmem [{local_classes[0]}~{local_classes[-1]}][{self.num_exemplars_per_class}]"
        prog_bar = tqdm(local_classes, desc=prog_bar_desc, position=rank)
        for class_id in prog_bar:
            # self.print_logger.debug(f'rank{rank} {class_id} in {local_classes}')
            inherent_class_id = self.data_manager.class_order[class_id].item()
            class_dataset = self.data_manager.get_dataset_by_class_ids([inherent_class_id], split='train', mode='test')
            
            class_features = self.extract_herding_features(class_dataset)
            class_mean = class_features.mean(dim=0, keepdim=True) # true class_mean of normed features
            
            class_selected_indices = []
            actual_idx_without_removal = torch.arange(len(class_features))
            selected_mean = torch.zeros((self.local_network.feature_dim,), device=self.device)
            for n in range(1, self.num_exemplars_per_class + 1):
                # self.print_logger.debug(f'[{class_id}/{num_classes}][{n}/{self.num_exemplars_per_class+1}] exemplars update memory')
                mu_p = ((n-1) * selected_mean + class_features) / n # try adding each sample feature to memory and compute mean
                idx = (mu_p - class_mean).norm(dim=-1).argmin().item() # compute the difference between the true class_mean, finding the best one.
                
                selected_mean = mu_p[idx]
                actual_idx = actual_idx_without_removal[idx].item()
                class_selected_indices.append(class_dataset.indices[actual_idx].item())

                mask = (torch.arange(len(class_features)) != idx)
                class_features = class_features[mask] # remove it to avoid duplicate selection
                actual_idx_without_removal = actual_idx_without_removal[mask] # remove it to avoid duplicate selection
            
            assert len(torch.unique(torch.tensor(class_selected_indices))) == len(class_selected_indices), "Duplicate Exemplars Selected."
            selected_indices.append(class_selected_indices)
            selected_mean /= selected_mean.norm()
            class_means.append(selected_mean)
        prog_bar.close()
        
        if self.distributed is not None:
            gathered_indices = self.distributed_gather_classes(
                torch.tensor(selected_indices, dtype=torch.int, device=self.device), 
                all_num_classes
            )

            gathered_class_means = self.distributed_gather_classes(
                torch.stack(class_means),
                all_num_classes
            )

            selected_indices = gathered_indices.tolist()
            class_means = [m for m in gathered_class_means]

            # self.print_logger.debug(f'selected_indices: {selected_indices}')
            # self.print_logger.debug(f'class_means: {class_means}')

        self.memory_samples.extend(selected_indices)
        self.class_means.extend(class_means)

    def state_dict(self) -> dict:
        super_dict = super().state_dict()
        return super_dict | {'class_means': self.class_means}
    
    def load_state_dict(self, d: dict):
        super().load_state_dict(d)
        self.class_means = d.get('class_means', [])
