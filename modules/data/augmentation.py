import torchvision.transforms as transforms
from .autoaugment import CIFAR10Policy, ImageNetPolicy, Cutout


cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

cifar100_train_1 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=63 / 255),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
])

cifar100_train_2 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=63 / 255),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
    transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
])

cifar100_test_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
])


imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

imagenet_train_1 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=63 / 255),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

imagenet_train_2 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=63 / 255),
    ImageNetPolicy(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

imagenet_test_1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

pil_none224 = transforms.Compose([
    transforms.Resize(224),
])

pil_none32 = transforms.Compose([
    transforms.Resize(32),
])


transform_dict = {
    'pil_none32': pil_none32,
    'pil_none224': pil_none224,
    'cifar100_train_1': cifar100_train_1,
    'cifar100_train_2': cifar100_train_2,
    'cifar100_test_1': cifar100_test_1,
    'imagenet_train_1': imagenet_train_1,
    'imagenet_train_2': imagenet_train_2,
    'imagenet_test_1': imagenet_test_1,
}

default_transform_dict = {
    'cifar': ('cifar100_train_1', 'cifar100_test_1'),
    'imagenet': ('imagenet_train_1', 'imagenet_test_1'),
    'cub': ('imagenet_train_1', 'imagenet_test_1'),
    'domainnet_dil': ('imagenet_train_1', 'imagenet_test_1'),
    'domainnet_cilood': ('imagenet_train_1', 'imagenet_test_1'),
}

def transform_dispatch(query: str):
    return transform_dict[query]
