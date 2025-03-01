from pathlib import Path
from argparse import ArgumentParser

def environment_arg(parser: ArgumentParser):
    # meta info
    parser.add_argument('--host-name', type=str, default='hostname')

def runtime_arg(parser: ArgumentParser):
    # device
    parser.add_argument('--device', type=str, default='cuda', help='support "cpu" and "cuda".')
    parser.add_argument('--dist-backend', default='nccl')

    # seed
    parser.add_argument('--seed', type=int, default=1993)

def train_runtime_arg(parser: ArgumentParser):
    # evaluation
    parser.add_argument('--eval-interval', type=int, default=1, help='eval interval in training by epochs.')

    # outputs
    parser.add_argument('--terminal-only', action='store_true', default=False, help='no side effect (log files, checkpoints, uploading) will be produced.')
    parser.add_argument('--disable-log-file', action='store_true', default=False)
    parser.add_argument('--disable-save-ckpt', action='store_true', default=False)
    parser.add_argument('--exp-name', type=str, default='demoexp')
    parser.add_argument('--exp-id', type=str, default='000000_000000')
    parser.add_argument('--ckpt-dir', type=Path, default='./checkpoints')
    parser.add_argument('--save-ckpt-tasks', type=list, default=[])
    parser.add_argument('--log-dir', type=Path, default='./logs')
    parser.add_argument('--output-file-prefix', type=str, default='exp')

    # inputs
    parser.add_argument('--ckpt-path', type=Path)
    parser.add_argument('--ckpt-task', type=int, help='start from 1.')

def scenario_arg(parser: ArgumentParser):
    # scenario
    parser.add_argument('--dataset-name', type=str, default='CIFAR100')
    parser.add_argument('--dataset-root', type=Path, default=Path('~/data/cifar/').expanduser())

    parser.add_argument('--scenario', default='CIL 10-10')
    parser.add_argument('--class-order')

    parser.add_argument('--ffcv', type=bool, default=False, help='use ffcv data loader')
    parser.add_argument('--train-beton-path', type=Path, default=Path('~/data/datasets/ffcv_beton/cifar100_train.beton').expanduser())
    parser.add_argument('--val-beton-path', type=Path, default=Path('~/data/datasets/ffcv_beton/cifar100_val.beton').expanduser())

def train_arg(parser: ArgumentParser):
    # experiment config
    parser.add_argument('--exp-configs', nargs='*', type=Path, help='experiment config, yaml config files, overwrites the argument values with same name', default=[])

    parser.add_argument('--method', type=str)

    # debug
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode, only train with several epochs')

def evaluate_arg(parser: ArgumentParser):
    # checkpoints
    parser.add_argument('--ckpt-paths', nargs='+')

def set_parser():
    parser = ArgumentParser(description='Continual Launcher', usage='')
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train', help='commence continual learning training')
    eval_parser = subparsers.add_parser('evaluate', help='commence continual learning evaluation')
    
    environment_arg(train_parser)
    runtime_arg(train_parser)
    train_runtime_arg(train_parser)
    scenario_arg(train_parser)
    train_arg(train_parser)
    
    environment_arg(eval_parser)
    runtime_arg(eval_parser)
    scenario_arg(eval_parser)
    evaluate_arg(eval_parser)


    return parser

def get_args(raw_args=None):
    parser = set_parser()
    args = parser.parse_args(raw_args)
    if args.terminal_only:
        args.disable_log_file = True
        args.disable_save_ckpt = True
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)
