from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ConstantLR


scheduler_dict = {
    'multistep': MultiStepLR,
    'cosineannealing': CosineAnnealingLR,
    'constant': ConstantLR,
}

default_scheduler = 'constant'

def scheduler_dispatch(optimizer, scheduler_configs: dict):
    scheduler_name = scheduler_configs.get('name', default_scheduler)
    Scheduler = scheduler_dict.get(scheduler_name.lower())

    if Scheduler is not None:
        scheduler_params = scheduler_configs.get('params', dict())
        scheduler = Scheduler(optimizer, **scheduler_params)
        return scheduler
    else:
        return None