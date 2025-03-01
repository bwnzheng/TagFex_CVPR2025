from pathlib import Path
from ffcv.loader import Loader, OrderOption

from ..augmentation import default_transform_dict
from .pipline import pipline_dispatch

def get_ffcv_loader(ds, beton_path, pipline_name, device, num_aug=1, **params):
    image_pipline, label_pipline = pipline_dispatch(pipline_name, device)

    if num_aug > 1:
        loader = Loader(
            Path(beton_path).expanduser(),
            indices=ds.indices,
            pipelines={
                'image': image_pipline, 
                'label': label_pipline, 
                'image2': image_pipline,
            },
            custom_field_mapper={'image2': 'image'},
            **params
        )
    else:
        loader = Loader(
            Path(beton_path).expanduser(),
            indices=ds.indices,
            pipelines={'image': image_pipline, 'label': label_pipline},
            custom_field_mapper={},
            **params
        )

    return loader

def get_ffcv_loaders(ds_train, ds_test, trainloader_params, testloader_params, device, configs, distributed):
    dataset_name = configs['dataset_name'].lower()
    train_pipline = configs.get('train_transform', default_transform_dict[dataset_name.strip('0123456789')][0])
    test_pipline = configs.get('test_transform', default_transform_dict[dataset_name.strip('0123456789')][1])

    train_loader = get_ffcv_loader(
        ds_train, Path(configs['train_beton_path']).expanduser(), train_pipline, device,
        order=OrderOption.RANDOM,
        seed=configs['seed'],
        distributed=distributed,
        num_aug=configs.get('num_aug', 1),
        **trainloader_params
    )
    
    test_loader = get_ffcv_loader(
        ds_test, Path(configs['val_beton_path']).expanduser(), test_pipline, device,
        order=OrderOption.SEQUENTIAL,
        seed=configs['seed'],
        distributed=distributed,
        # drop_last=False,
        **testloader_params
    )

    return train_loader, test_loader