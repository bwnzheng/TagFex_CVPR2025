import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import HerdingIndicesLearner
from .dernet import DERNet
from modules import Accuracy, MeanMetric, CatMetric, select_metrics, forward_metrics, get_metrics
from modules import optimizer_dispatch, scheduler_dispatch, get_loaders
from utils.configuration import load_configs
from utils.funcs import parameter_count

from loggers import LoguruLogger, loguru

EPSILON = 1e-8

class DER(HerdingIndicesLearner):
    def __init__(self, data_maganger, configs: dict, device, distributed=False) -> None:
        super().__init__(data_maganger, configs, device, distributed)

        self._init_network(self.configs.get('backbone_configs', dict()), self.configs.get('network_configs', dict()))
        
        if self.distributed is not None:
            self._init_ddp()
        
        self._init_loggers()
        self.print_logger.info(configs)

        self.print_logger.info(f'class order: {self.data_manager.class_order.tolist()}')
        self.ordered_index_map = torch.from_numpy(self.data_manager.ordered_index_map).to(self.device)
    
    def _init_network(self, backbone_configs, network_configs):
        if backbone_configs['name'] == 'resnet18':
            params = {
                'dataset_name': self.data_manager.dataset_name, 
                'small_base': (self.data_manager.init_num_cls == self.data_manager.inc_num_cls)
            }
            backbone_configs.update(params=params)
        self.network = DERNet(backbone_configs, network_configs, self.device)
        self.local_network = self.network # for ddp compatibility
    
    def _init_ddp(self):
        self.configs['trainloader_params']['batch_size'] //= self.distributed['world_size']
        torch.distributed.barrier()

    def _model_to_ddp(self):
        self.network = nn.parallel.DistributedDataParallel(
            self.local_network, device_ids=[self.distributed['rank']], find_unused_parameters=self.configs['debug']
        )

    @torch.no_grad()
    def extract_herding_features(self, dataset: torch.utils.data.Dataset) -> torch.Tensor:
        self.print_logger.debug(f'dataset length: {len(dataset.indices)}')
        if self.configs['ffcv']:
            from modules.data.ffcv.loader import get_ffcv_loader, default_transform_dict, OrderOption
            dataset_name = self.configs['dataset_name'].lower()
            pipline_name = self.configs.get('test_transform', default_transform_dict[dataset_name.strip('0123456789')][0])
            data_loader = get_ffcv_loader(
                dataset, self.configs['train_beton_path'], pipline_name, self.device,
                order=OrderOption.SEQUENTIAL,
                seed=self.configs['seed'],
                drop_last=False,
                **self.configs['testloader_params']
            )
        else:
            data_loader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.configs['testloader_params']['batch_size'],
                num_workers=0 # extremely slow when setting to >0
            )
        
        self.print_logger.debug(f'dataloader length: {len(data_loader)}')
        self.local_network.eval()
        features = []
        for samples, targets in data_loader:
            samples = samples.to(self.device, non_blocking=True)
            feats = self.local_network(samples.contiguous())["concat_feature"]
            features.append(feats)
        features = torch.cat(features)
        features /= features.norm(dim=-1, keepdim=True) + EPSILON
        return features
    
    @loguru.logger.catch
    def train(self) -> None:
        # >>> @train_start
        self.update_state(run_state='train', num_tasks=self.data_manager.num_tasks)
        self.run_metrics = {
            'acc1_curve': CatMetric(sync_on_compute=False).to(self.device),
            'acc5_curve': CatMetric(sync_on_compute=False).to(self.device),
            'nme1_curve': CatMetric(sync_on_compute=False).to(self.device),
            'nme5_curve': CatMetric(sync_on_compute=False).to(self.device),
            'avg_acc1': MeanMetric().to(self.device),
            'avg_acc5': MeanMetric().to(self.device),
            'avg_nme1': MeanMetric().to(self.device),
            'avg_nme5': MeanMetric().to(self.device)
        }
        # <<< @train_start

        for task_id, (task_train, task_test) in enumerate(self.data_manager.tasks):
            # >>> @train_task_start
            cur_task_num_classes = self.data_manager.task_num_cls[task_id]
            sofar_num_classes = sum(self.data_manager.task_num_cls[:task_id+1])
            learned_num_classes = sofar_num_classes - cur_task_num_classes
            self.update_state(cur_task=task_id+1, cur_task_num_classes=cur_task_num_classes, sofar_num_classes=sofar_num_classes, learned_num_classes=learned_num_classes)

            self.local_network.update_network(cur_task_num_classes)
            self.local_network.freeze_old_backbones()
            if self.configs['ckpt_path'] is not None and self.configs['ckpt_task'] is not None and task_id + 1 <= self.configs['ckpt_task']:
                if task_id + 1 == self.configs['ckpt_task']:
                    self._load_checkpoint(self.configs['ckpt_path'])
                continue
            
            total, trainable = parameter_count(self.local_network)
            self.print_logger.info(f'{self._get_status()} | parameters: {total} in total, {trainable} trainable.')
            
            if self.distributed is not None:
                self._model_to_ddp()
            
            # add memory into training set.
            memory_indices = self.get_memory()
            new_indices = np.concatenate((task_train.indices, memory_indices))
            task_train.indices = new_indices

            # get dataloaders
            if self.configs['ffcv']:
                from modules.data.ffcv.loader import get_ffcv_loaders
                train_loader, test_loader = get_ffcv_loaders(task_train, task_test, self.configs['trainloader_params'], self.configs['testloader_params'], self.device, self.configs, self.distributed is not None)
            else:
                train_loader, test_loader = get_loaders(task_train, task_test, self.configs['trainloader_params'], self.configs['testloader_params'], self.distributed)
            # <<< @train_task_start

            self.train_task(train_loader, test_loader)

            # >>> @train_task_end
            self.print_logger.info(f"Adjust memory to {self.num_exemplars_per_class} per class ({self.num_exemplars_per_class * self.state['sofar_num_classes']} in total).")
            self.reduce_memory()
            self.update_memory()

            if task_id > 0:
                self.print_logger.info('Weight align before task end evaluation.')
                self.local_network.weight_align(cur_task_num_classes)
            
            # evaluation as task end
            results = self.eval_epoch(test_loader)
            forward_metrics(select_metrics(self.run_metrics, 'acc1'), results['eval_acc1'])
            forward_metrics(select_metrics(self.run_metrics, 'acc5'), results['eval_acc5'])
            forward_metrics(select_metrics(self.run_metrics, 'nme1'), results['eval_nme1'])
            forward_metrics(select_metrics(self.run_metrics, 'nme5'), results['eval_nme5'])
            self.print_logger.success(f'{self._get_status()}\n├> {self._metric_repr(results | get_metrics(self.run_metrics))}')
            if self.configs['ckpt_dir'] is not None and not self.configs['disable_save_ckpt'] and task_id + 1 in self.configs['save_ckpt_tasks']:
                self._save_checkpoint()
            # <<< @train_task_end
        
        # >>> @train_end
        results = get_metrics(self.run_metrics)
        self.print_logger.success(f'{self._get_status()}\n├> {self._metric_repr(results)}')
        self.update_state(run_state='finished')
        # <<< @train_end

    def train_task(self, train_loader, test_loader):
        # get optimizer and lr scheduler
        if self.state['cur_task'] == 1:
            optimizer = optimizer_dispatch(self.local_network.parameters(), self.configs['init_optimizer_configs'])
            scheduler = scheduler_dispatch(optimizer, self.configs['init_scheduler_configs'])

            num_epochs = self.configs['init_epochs'] # FIXME: config category
        else:
            optimizer = optimizer_dispatch(self.local_network.parameters(), self.configs['inc_optimizer_configs'])
            scheduler = scheduler_dispatch(optimizer, self.configs['inc_scheduler_configs'])

            num_epochs = self.configs['inc_epochs'] # FIXME: config category

        # >>> @after_train_task_setups
        if self.configs['debug']:
            num_epochs = 5
        self.update_state(cur_task_num_epochs=num_epochs)
        # <<< @after_train_task_setups

        rank = 0 if self.distributed is None else self.distributed['rank']
        prog_bar = tqdm(range(num_epochs), desc=f"Task {self.state['cur_task']}/{self.state['num_tasks']}") if rank == 0 else range(num_epochs)
        for epoch in prog_bar:
            # >>> @train_epoch_start
            self.update_state(cur_epoch=epoch + 1, num_batches=len(train_loader))
            self.add_state(accumulated_cur_epoch=1)
            if self.distributed is not None and not self.configs['ffcv']:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)
            # <<< @train_epoch_start

            train_results = self.train_epoch(train_loader, optimizer, scheduler)
            
            # >>> @train_epoch_end
            if epoch % self.configs['eval_interval'] == 0:
                eval_results = self.eval_epoch(test_loader)
                self.print_logger.info(f'{self._get_status()} | {self._metric_repr(train_results)} {self._metric_repr(eval_results)}')
            else:
                self.print_logger.info(f'{self._get_status()} | {self._metric_repr(train_results)}')
            # <<< @train_epoch_end
        if rank == 0:
            prog_bar.close()

    def train_epoch(self, train_loader, optimizer, scheduler):
        self.network.train()
        num_classes = self.state['sofar_num_classes'].item()
        metrics = {
            'loss': MeanMetric().to(self.device),
            'cls_loss': MeanMetric().to(self.device),
            'train_acc1': Accuracy(task='multiclass', num_classes=num_classes).to(self.device)
        }
        if self.state['cur_task'] > 1:
            metrics.update(aux_loss=MeanMetric().to(self.device))
        
        for batch, batch_data in enumerate(train_loader):
            batch_data = tuple(data.to(self.device, non_blocking=True) for data in batch_data)

            samples, targets = batch_data
            targets = self.ordered_index_map[targets.flatten()] # map to continual class id.
            # self.print_logger.debug(f'train {batch}/{len(train_loader)}', samples.device, targets.device)
            # self.print_logger.debug(f'batch shape {samples.shape}')

            # >>> @train_batch_start
            self.update_state(cur_batch=batch+1)
            # <<< @train_batch_start

            # >>> @train_forward
            out = self.network(samples.contiguous())
            logits = out["logits"]
            cls_loss = F.cross_entropy(logits, targets)

            if (aux_logits := out.get('aux_logits')) is not None:
                learned_num_classes = self.state['learned_num_classes']
                aux_targets = torch.where(
                    targets >= learned_num_classes,
                    targets - learned_num_classes + 1,
                    0
                )
                self.print_logger.debug(f'targets: {targets}')
                self.print_logger.debug(f'aux_targets: {aux_targets}')
                self.print_logger.debug(f'aux_logits: {aux_logits}')
                aux_loss = F.cross_entropy(aux_logits, aux_targets)
                loss = cls_loss + aux_loss
            else:
                loss = cls_loss
            # <<< @train_forward

            # >>> @train_backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # <<< @train_backward

            # >>> @train_batch_end
            metrics['loss'].update(loss.detach())
            metrics['cls_loss'].update(cls_loss.detach())
            metrics['train_acc1'].update(logits.detach(), targets.detach())
            if self.state['cur_task'] > 1:
                metrics['aux_loss'].update(aux_loss.detach())

            # <<< @train_batch_end
        
        scheduler.step()
        train_results = get_metrics(metrics)
        return train_results

    @torch.no_grad()
    def eval_epoch(self, data_loader):
        # >>> @eval_start
        prev_run_state = self.state.get('run_state')
        self.update_state(run_state='eval')
        self.network.eval()
        # <<< @eval_start

        num_classes = self.state['sofar_num_classes'].item()
        acc_metrics = {
            'eval_acc1': Accuracy(task='multiclass', num_classes=num_classes).to(self.device),
            'eval_acc5': Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(self.device),
            'eval_acc1_per_class': Accuracy(task='multiclass', average=None, num_classes=num_classes).to(self.device),
            'eval_acc5_per_class': Accuracy(task='multiclass', average=None, num_classes=num_classes, top_k=5).to(self.device),
        }
        if len(self.class_means) == self.state['sofar_num_classes']:
            nme_metrics = {
                'eval_nme1': Accuracy(task='multiclass', num_classes=num_classes).to(self.device),
                'eval_nme5': Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(self.device),
                'eval_nme1_per_class': Accuracy(task='multiclass', average=None, num_classes=num_classes).to(self.device),
                'eval_nme5_per_class': Accuracy(task='multiclass', average=None, num_classes=num_classes, top_k=5).to(self.device),
            }

        # >>> @eval_epoch_start
        self.update_state(eval_num_batches=len(data_loader))
        # <<< @eval_epoch_start

        for batch, batch_data in enumerate(data_loader):
            batch_data = tuple(data.to(self.device, non_blocking=True) for data in batch_data)

            # >>> @eval_batch_start
            self.update_state(eval_cur_batch=batch+1)
            # <<< @eval_batch_start

            samples, targets = batch_data
            targets = self.ordered_index_map[targets.flatten()] # map to continual class id.
            # self.print_logger.debug(f'eval {batch}/{len(data_loader)}', samples.device, targets.device)
            outs = self.network(samples.contiguous())
            logits = outs['logits']
            forward_metrics(acc_metrics, logits, targets)

            if len(self.class_means) == self.state['sofar_num_classes']:
                features = outs['concat_feature']
                features /= features.norm(dim=-1, keepdim=True) + EPSILON
                dists = torch.cdist(features, torch.stack(self.class_means))
                forward_metrics(nme_metrics, -dists, targets)

        acc_metric_results = get_metrics(acc_metrics)
        if len(self.class_means) == self.state['sofar_num_classes']:
            nme_metric_results = get_metrics(nme_metrics)
        else:
            nme_metric_results = dict()

        # >>> @eval_end
        self.update_state(run_state=prev_run_state)
        # >>> @eval_end

        return acc_metric_results | nme_metric_results
    
    def _metric_repr(self, metric_results: dict):
        def merge_to_task(acc_per_cls):
            display_value = []
            accumuated_num_cls = 0
            for num_cls in self.data_manager.task_num_cls:
                tot_num_cls = accumuated_num_cls + num_cls
                if tot_num_cls > len(acc_per_cls):
                    break
                task_acc = acc_per_cls[accumuated_num_cls:tot_num_cls].mean()
                display_value.append(task_acc.item())
                accumuated_num_cls = tot_num_cls
            return display_value

        scalars = []
        vectors = []
        for key, value in metric_results.items():
            if 'acc' in key or 'nme' in key:
                display_value = value * 100
            else:
                display_value = value
            
            if value.dim() > 0:
                if 'per_class' in key:
                    display_value = merge_to_task(display_value)
                    key = key.replace('per_class', 'per_task')
                else:
                    display_value = display_value.cpu().tolist()
                [f"{v:.2f}" for v in display_value]
                r = f'{key} [{" ".join([f"{v:.2f}" for v in display_value])}]'
                vectors.append(r)
            else:
                r = f'{key} {display_value.item():.2f}'
                scalars.append(r)
        
        componets = []
        if len(scalars) > 0:
            componets.append(' '.join(scalars))
        if len(vectors) > 0:
            componets.append('\n├> '.join(vectors))
        s = '\n├> '.join(componets)
        return '\n└>'.join(s.rsplit('\n├>', 1))

    def _init_loggers(self):
        self.loguru_logger = LoguruLogger(self.configs, self.configs['disable_log_file'], tqdm_out=True)
        self.print_logger = self.loguru_logger.logger # the actual logger
    
    def _get_status(self):
        if self.distributed is None:
            rank, world_size = 0, 1
        else:
            rank, world_size = self.distributed['rank'], self.distributed['world_size']
        run_state = self.state.get('run_state')
        num_tasks = self.state.get('num_tasks')
        cur_task = self.state.get('cur_task')
        cur_task_num_classes = self.state.get('cur_task_num_classes')
        sofar_num_classes = self.state.get('sofar_num_classes')
        cur_task_num_epochs = self.state.get('cur_task_num_epochs')
        cur_epoch = self.state.get('cur_epoch')
        num_batches = self.state.get('num_batches')
        cur_batch = self.state.get('cur_batch')
        eval_num_batches = self.state.get('eval_num_batches')
        eval_cur_batch = self.state.get('eval_cur_batch')

        if run_state == 'train':
            status = f"R{rank}T[{cur_task}/{num_tasks}]E[{cur_epoch}/{cur_task_num_epochs}] {run_state}"
        elif run_state == 'eval':
            status = f"R{rank}T[{cur_task}/{num_tasks}]E[{cur_epoch}/{cur_task_num_epochs}] {run_state}"
        
        return status

    def state_dict(self) -> dict:
        super_dict = super().state_dict()
        d = {
            "network_state_dict": self.local_network.state_dict(),
            "run_metrics": {name: metric.state_dict() for name, metric in self.run_metrics.items()}
        }
        return super_dict | d
    
    def load_state_dict(self, d) -> None:
        super().load_state_dict(d)
        network_state_dict = d['network_state_dict']
        self.local_network.load_state_dict(network_state_dict)
        
        run_metrics = d.get('run_metrics', dict())
        for name, state_dict in run_metrics.items():
            self.run_metrics[name].load_state_dict(state_dict)

    def _save_checkpoint(self):
        save_dict = self.state_dict()

        cur_task = self.state['cur_task']
        num_tasks = self.state['num_tasks']
        dataset_name = self.data_manager.dataset_name
        task_name, scenario = self.data_manager.scenario.split(' ')
        method = self.configs['method']

        ckpt_file_name = f"{method}_{dataset_name}_{scenario}_[{cur_task}_{num_tasks}].ckpt"
        ckpt_dir = self.configs['ckpt_dir']
        ckpt_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        torch.save(save_dict, ckpt_dir / ckpt_file_name)
    
    def _load_checkpoint(self, path):
        ckpt = torch.load(path, self.device)
        self.load_state_dict(ckpt)