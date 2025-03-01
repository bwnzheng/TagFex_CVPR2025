import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from functools import cache

from .dataset import get_dataset
from .augmentation import default_transform_dict

class ContinualDataManager:
    def __init__(self, configs, distributed) -> None:
        self.configs = configs
        self.distributed = distributed
        self._init_properties()

        self.dataset_train = self._get_dataset(split='train', mode='train')
        self.dataset_test = self._get_dataset(split='test', mode='test')

        self.total_num_cls = len(self.dataset_train.classes)
        self._scenario_dispatch(self.scenario)
    
    @cache
    def _get_dataset(self, split='train', mode='train'):
        if mode == 'train':
            transform = self.train_transform
        elif mode == 'test':
            transform = self.test_transform
        else:
            raise ValueError(f'Unknown mode {mode}.')
        
        dataset = get_dataset(
            self.dataset_name,
            split=split,
            transform=transform,
            num_aug=self.configs.get('num_aug', 1) if mode == 'train' else 1,
            root=self.dataset_root
        )
        return dataset

    def _init_properties(self):
        self.dataset_name = self.configs['dataset_name'].lower()
        self.dataset_root = self.configs['dataset_root']
        self.train_transform = self.configs.get('train_transform', default_transform_dict[self.dataset_name.strip('0123456789')][0])
        self.test_transform = self.configs.get('test_transform', default_transform_dict[self.dataset_name.strip('0123456789')][1])

        self.scenario = self.configs['scenario']

    def _scenario_dispatch(self, scenario_name: str):
        scenario_name = scenario_name.lower()
        self.task_name, scenario = scenario_name.split(' ')
        if self.task_name == 'cil':
            if scenario == 'joint':
                self.init_num_cls, self.inc_num_cls = self.total_num_cls, self.total_num_cls
            else:
                init_num_cls, inc_num_cls = scenario.split('-')
                self.init_num_cls, self.inc_num_cls = int(init_num_cls), int(inc_num_cls)

            if (self.total_num_cls - self.init_num_cls) % self.inc_num_cls != 0:
                raise ValueError(f'Invalid CIL scenario {scenario}.')
            
            self.task_num_cls = [self.init_num_cls] + [self.inc_num_cls] * ((self.total_num_cls - self.init_num_cls) // self.inc_num_cls)
            self.task_num_cls = np.array(self.task_num_cls)
            self.num_tasks = len(self.task_num_cls)

            self.class_order = self.configs.get('class_order') # reordered class ids -> inherent class ids
            if self.class_order is not None:
                self.class_order = np.array(self.class_order)
            elif self.distributed is None:
                self.class_order = np.random.permutation(np.arange(self.total_num_cls))
            else: 
                class_order = torch.randperm(self.total_num_cls, device='cuda')
                torch.distributed.broadcast(class_order, src=0)
                self.class_order = class_order.cpu().numpy()
            
            # self.dataset_train.targets = _map_new_class_index(self.dataset_train.targets, self.class_order.tolist())
            # self.dataset_test.targets = _map_new_class_index(self.dataset_test.targets, self.class_order.tolist())

            self.ordered_index_map = _map_new_class_index(np.arange(self.total_num_cls), self.class_order.tolist()) # inherent class ids -> class ids reordered by $class_order
        elif self.task_name == 'dil':
            self.num_domains = self.dataset_train.num_domains
        else:
            raise ValueError(f'Unknown task name {self.task_name}.')

    def get_dataset_by_class_ids(self, class_ids, split='train', mode='train'):
        """
        Get dataset by inherent class ids.
        """
        class_ids = np.array(class_ids)
        
        base_dataset = self._get_dataset(split=split, mode=mode)
        indices = np.argwhere(np.isin(base_dataset.targets, class_ids)).flatten()
        
        dataset = Subset(base_dataset, indices)
        return dataset
    
    def get_dataset_by_indices(self, indices, split='train', mode='train'):
        """
        Get dataset by indices in the split.
        """
        indices = np.array(indices)
        base_dataset = self._get_dataset(split=split, mode=mode)
        
        dataset = Subset(base_dataset, indices)
        return dataset

    @cache
    def get_class_names(self):
        class_list = []
        for c in self.dataset_train.classes:
            if isinstance(c, tuple):
                class_list.append(c[0])
            else:
                class_list.append(c)
        return class_list
    
    @cache
    def get_class_names_by_task(self, task_id, mode):
        num_prev_classes = sum(self.task_num_cls[:task_id])
        num_classes = self.task_num_cls[task_id]
        if mode == 'train':
            class_ids = self.class_order[num_prev_classes:num_classes]
        elif mode == 'test':
            class_ids = self.class_order[:num_prev_classes+num_classes]
        class_names_list_all = self.get_class_names()
        class_names = np.array(class_names_list_all)[class_ids]
        return class_names.tolist()

    @property
    def tasks(self):
        if self.task_name == 'cil':
            return self._cil_tasks()
        elif self.task_name == 'dil':
            return self._dil_tasks()
    
    def _cil_tasks(self):
        num_prev_classes = 0
        for num_classes in self.task_num_cls:
            train_class_ids = self.class_order[num_prev_classes:num_prev_classes+num_classes] # first $num_classes classes in $class_order
            task_train = self.get_dataset_by_class_ids(train_class_ids, split='train', mode='train')

            test_class_ids = self.class_order[:num_prev_classes+num_classes]
            task_test = self.get_dataset_by_class_ids(test_class_ids, split='test', mode='test')

            yield task_train, task_test

            num_prev_classes += num_classes
    
    def _dil_tasks(self):
        for domain_id in range(self.num_domains):
            train_domain_indices = self.dataset_train.get_domain_indices(domain_id)
            task_train = Subset(self.dataset_train, train_domain_indices)

            test_domain_indices = self.dataset_test.get_domain_indices(domain_id)
            task_test = Subset(self.dataset_test, test_domain_indices)

            yield task_train, task_test



def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

if __name__ == '__main__':
    test_config = {
        'dataset_name': 'cifar100',
        'dataset_root': '~/data/cifar',
        'scenario': 'CIL 10-10',
    }
    data_manager = ContinualDataManager(test_config)
    for task_train, task_test in data_manager.tasks:
        print(len(task_train), len(task_test))