import os, datetime
import torch
import utils.funcs as utlf
from utils.configuration import load_configs, load_yaml

from modules.data.manager import ContinualDataManager
from modules.learner.base import ContinualLearner
from methods import method_dispatch

class ContinualLauncher:
    def __init__(self, args) -> None:
        self.args = args

        self._init_torch_dist()

        if self.distributed is not None:
            self.device = torch.device(f'{args.device}:{self.distributed["rank"]}')
            utlf.set_seed(args.seed + self.distributed['rank'])
        else:
            self.device = torch.device(args.device)
            utlf.set_seed(args.seed)

    def _get_train_configs(self):
        exp_configs = load_configs(self.args.exp_configs)

        self.configs = vars(self.args)
        self.configs.update(**exp_configs)

    def train(self):
        self._get_train_configs()
        data_manager = ContinualDataManager(self.configs, self.distributed)
        
        learner: ContinualLearner = method_dispatch(self.configs['method'], data_manager, self.configs, self.device, self.distributed)

        learner.train()

    def _get_evaluate_configs(self):
        exp_configs = load_configs(self.args.exp_configs)

        self.configs = vars(self.args)
        self.configs.update(**exp_configs)

    def evaluate(self):
        pass

    def reconstruct(self):
        pass

    def _init_torch_dist(self):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.distributed = dict()
            self.distributed['rank'] = int(os.environ['RANK'])
            self.distributed['world_size'] = int(os.environ['WORLD_SIZE'])
            self.distributed['local_rank'] = int(os.environ['LOCAL_RANK'])
        else:
            self.distributed = None
            return

        import multiprocessing
        multiprocessing.set_start_method('spawn')

        torch.cuda.set_device(self.distributed['local_rank'])
        torch.distributed.init_process_group(
            init_method='env://',
            backend=self.args.dist_backend, 
            world_size=self.distributed['world_size'], 
            rank=self.distributed['rank'],
            # timeout=datetime.timedelta(seconds=5)
        )
        torch.distributed.barrier()

if __name__ == '__main__':
    from utils.argument import get_args
    args = get_args()
    launcher = ContinualLauncher(args)

    if args.command == 'train':
        launcher.train()
    elif args.command == 'evaluate':
        launcher.evaluate()
    else:
        raise ValueError(f'Unknown command {args.command}')
