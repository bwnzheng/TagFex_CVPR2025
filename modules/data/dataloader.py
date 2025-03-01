from torch.utils.data import DataLoader, DistributedSampler

def get_loaders(ds_train, ds_test, trainloader_params, testloader_params, distributed=None):
    if distributed is None:
        train_loader = DataLoader(
            ds_train,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            **trainloader_params
        )
        test_loader = DataLoader(
            ds_test,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True,
            **testloader_params
        )
    else:
        rank = distributed['rank']
        world_size = distributed['world_size']

        train_sampler = DistributedSampler(
            ds_train, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True, 
            drop_last=True
        )
        train_loader = DataLoader(
            ds_train,
            persistent_workers=True,
            **trainloader_params,
            sampler=train_sampler,
        )
        # if len(ds_test) % world_size != 0:
        #     raise RuntimeWarning(f'World size ({world_size}) does not divide number of test samples ({len(ds_test)}).')

        test_sampler = DistributedSampler(
            ds_test, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False
        )

        test_loader = DataLoader(
            ds_test,
            persistent_workers=True,
            **testloader_params,
            sampler=test_sampler,
        )

    return train_loader, test_loader


    
