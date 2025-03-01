import random, os
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parameter_count(module: torch.nn.Module):
    total_count = sum(p.numel() for p in module.parameters())
    trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_count, trainable_count
