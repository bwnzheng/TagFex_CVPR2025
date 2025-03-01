from torch.optim import SGD, Adam, AdamW


optimizer_dict = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
}

default_optimizer = 'sgd'

def optimizer_dispatch(parameters, optimizer_configs: dict):
    optimizer_name = optimizer_configs.get('name', default_optimizer)
    optimizer = optimizer_dict.get(optimizer_name)

    if optimizer is not None:
        optimizer_params = optimizer_configs.get('params', dict())
        return optimizer(parameters, **optimizer_params)
    else:
        return None