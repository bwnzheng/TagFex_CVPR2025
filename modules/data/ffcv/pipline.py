import torch

import torchvision.transforms as transforms
import ffcv.transforms as fftransforms
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder
from .operation import DivideImage255

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def pipline_dispatch(query: str, device):
    if query == 'cifar100_train_1':
        pipline = [
            # RandomResizedCropRGBImageDecoder(output_size=(32,32), scale=(1.,1.), ratio=(1.,1.)),
            SimpleRGBImageDecoder(),
            fftransforms.RandomTranslate(padding=4),
            fftransforms.RandomHorizontalFlip(flip_prob=.5),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
        ]
    elif query == 'cifar100_train_2':
        pipline = [
            # RandomResizedCropRGBImageDecoder((32,32)),
            SimpleRGBImageDecoder(),
            fftransforms.RandomTranslate(padding=4),
            fftransforms.RandomHorizontalFlip(flip_prob=.5),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            fftransforms.Cutout(16),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        ]
    elif query == 'cifar100_test_1':
        pipline = [
            SimpleRGBImageDecoder(),
            fftransforms.RandomHorizontalFlip(flip_prob=0.), # no-op, to avoid a bug in FFCV-SSL.
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
        ]
    elif query == 'imagenet_train_1':
        pipline = [
            RandomResizedCropRGBImageDecoder((224, 224)),
            fftransforms.RandomHorizontalFlip(),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
        ]
    elif query == 'imagenet_train_2':
        pipline = [
            RandomResizedCropRGBImageDecoder((224, 224)),
            fftransforms.RandomHorizontalFlip(),
            fftransforms.RandomColorJitter(brightness=63 / 255),
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        ]
    elif query == 'imagenet_test_1':
        pipline = [
            RandomResizedCropRGBImageDecoder((224, 224), (1,1), (1,1)),
            fftransforms.RandomHorizontalFlip(flip_prob=0.), # no-op, to avoid a bug in FFCV-SSL.
            fftransforms.ToTensor(),
            fftransforms.ToTorchImage()
        ]
    
    pipline.extend([
        fftransforms.ToDevice(torch.device(device)),
        fftransforms.Convert(torch.float32),
        DivideImage255(),
    ])

    if query.startswith('cifar100'):
        # pipline.append(fftransforms.NormalizeImage(mean=np.array(cifar_mean), std=np.array(cifar_std), type=np.float32))
        pipline.append(transforms.Normalize(mean=cifar100_mean, std=cifar100_std, inplace=True))
    elif query.startswith('imagenet'):
        # pipline.append(fftransforms.NormalizeImage(mean=np.array(imagenet_mean), std=np.array(imagenet_std), type=np.float32))
        pipline.append(transforms.Normalize(mean=imagenet_mean, std=imagenet_std, inplace=True))
    
    label_pipeline = [
        IntDecoder(), 
        fftransforms.ToTensor(),
        fftransforms.ToDevice(torch.device(device))
    ]

    return pipline, label_pipeline
