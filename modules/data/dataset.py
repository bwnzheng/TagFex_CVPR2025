import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder, VisionDataset
from PIL import Image

from .augmentation import transform_dispatch


class ImageNet100(ImageFolder):
    def __init__(self, root: str, split: str = 'train', **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = split
        super().__init__(self.split_folder, **kwargs)
        self.root = root

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

class CUB200(VisionDataset):
    def __init__(self, root: str, train: bool = True, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        self.is_train = train
        with open(os.path.join(self.root, 'images.txt')) as f:
            self.img_name_list = [line[:-1].split(' ')[-1] for line in f]
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            self.label_list = [int(line[:-1].split(' ')[-1]) - 1 for line in f]
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            self.train_test_list = [int(line[:-1].split(' ')[-1]) for line in f]
        with open(os.path.join(self.root, 'classes.txt')) as f:
            self.classes = [line[:-1].split(' ')[-1] for line in f]
        
        self.train_file_list = [name for i, name in zip(self.train_test_list, self.img_name_list) if i]
        self.test_file_list = [name for i, name in zip(self.train_test_list, self.img_name_list) if not i]

        self.train_label_list = [label for i, label in zip(self.train_test_list, self.label_list) if i]
        self.test_label_list = [label for i, label in zip(self.train_test_list, self.label_list) if not i]

        if self.is_train:
            self.samples = [(name, label) for name, label in zip(self.train_file_list, self.train_label_list)]
            self.targets = np.array(self.train_label_list)
        else:
            self.samples = [(name, label) for name, label in zip(self.test_file_list, self.test_label_list)]
            self.targets = np.array(self.test_label_list)

    def __getitem__(self, index: int):
        name, target = self.samples[index]
        sample = pil_loader(os.path.join(self.root, 'images', name))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

    def __len__(self):
        return len(self.samples)


class BaseDomainNet(VisionDataset):
    domain_names = ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch')
    def __init__(self, root: str, train: bool = True, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        self.is_train = train
        self.split = 'train' if self.is_train else 'test'
        self.classes = sorted(os.listdir(os.path.join(self.root, self.domain_names[0])))

    def __getitem__(self, index):
        name, target = self.samples[index]
        sample = pil_loader(os.path.join(self.root, name))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _read_split_file(self, path):
        file_list = []
        label_list = []
        with open(path) as f:
            for line in f:
                path, label = line[:-1].split(' ')
                file_list.append(path)
                label_list.append(int(label))
        
        return file_list, label_list

    @property
    def num_domains(self):
        return len(self.domain_names)

class DomainNetDIL(BaseDomainNet):
    def __init__(self, root, train = True, transforms=None, transform=None, target_transform=None):
        super().__init__(root, train, transforms, transform, target_transform)

        self.file_list = []
        self.label_list = []
        self.samples = []
        self.targets = []
        for name in self.domain_names:
            domain_file_list, domain_label_list = self._read_split_file(os.path.join(self.root, 'splits', f'{name}_{self.split}.txt'))
            self.file_list.append(domain_file_list)
            self.label_list.append(domain_label_list)

            domain_samples = list(zip(domain_file_list, domain_label_list))
            self.samples.extend(domain_samples)
            self.targets.extend(domain_label_list)

        self.targets = np.array(self.targets)
    
    def get_domain_indices(self, domain_id):
        domain_num_samples = [len(l) for l in self.file_list]
        st = sum(domain_num_samples[:domain_id])
        ed = sum(domain_num_samples[:domain_id+1])
        return np.arange(st, ed)

class DomainNetCILOOD(BaseDomainNet):
    def __init__(self, root, train = True, transforms=None, transform=None, target_transform=None):
        super().__init__(root, train, transforms, transform, target_transform)

        self.file_list = []
        self.label_list = []
        self.samples = []
        self.targets = []
        
        active_domains = self.domain_names[:-1] if train else self.domain_names[-1:]
        for name in active_domains:
            train_domain_file_list, train_domain_label_list = self._read_split_file(os.path.join(self.root, 'splits', f'{name}_train.txt'))
            test_domain_file_list, test_domain_label_list = self._read_split_file(os.path.join(self.root, 'splits', f'{name}_test.txt'))

            domain_file_list = train_domain_file_list + test_domain_file_list
            domain_label_list = train_domain_label_list + test_domain_label_list

            self.file_list.append(domain_file_list)
            self.label_list.append(domain_label_list)

            domain_samples = list(zip(domain_file_list, domain_label_list))
            self.samples.extend(domain_samples)
            self.targets.extend(domain_label_list)
        
        self.targets = np.array(self.targets)

class MultipleAugmentationDataset(Dataset):
    def __init__(self, base_dataset, num_aug=2) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.num_aug = num_aug

    def __getitem__(self, index):
        sample, target = self.base_dataset[index]
        samples = [self.base_dataset[index][0] for _ in range(self.num_aug - 1)]
        return sample, *samples, target
    
    def __getattr__(self, name):
        if name in ('targets', 'classes', 'num_domains'):
            return getattr(self.base_dataset, name)
        else:
            raise AttributeError(f'No attribute called {name}.')

def pil_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

dataset_name_map = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'imagenet1000': ImageNet,
    'imagenet100': ImageNet100,
    'cub200': CUB200,
    'domainnet_dil': DomainNetDIL,
    'domainnet_cilood': DomainNetCILOOD,
}


def get_dataset(name: str, split, transform, num_aug=1, **dataset_args):
    name = name.lower()
    DS = dataset_name_map[name]

    if name.startswith('cifar') or name.startswith('cub') or name.startswith('domainnet'):
        is_train = (split == 'train')
        dataset: VisionDataset = DS(train=is_train, transform=transform_dispatch(transform), **dataset_args)
    elif name.startswith('imagenet'):
        split = 'val' if split == 'test' else split
        dataset: VisionDataset = DS(split=split, transform=transform_dispatch(transform), **dataset_args)
    
    if split == 'train' and num_aug > 1:
        dataset = MultipleAugmentationDataset(dataset, num_aug)

    return dataset
