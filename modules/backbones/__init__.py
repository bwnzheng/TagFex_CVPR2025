from .cifar_resnet import resnet32, resnet20
from .resnet import resnet18, resnet10


backbone_dict = {
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet10': resnet10,
    'resnet18': resnet18,
}

default_backbone = 'resnet32'

def backbone_dispatch(backbone_configs: dict):
    backbone_name = backbone_configs.get('name', default_backbone)
    backbone = backbone_dict.get(backbone_name)

    if backbone is not None:
        backbone_params = backbone_configs.get('params', dict())
        return backbone(**backbone_params)
    else:
        return None
    
def register_backbone(**kwargs):
    backbone_dict.update(**kwargs)
