from typing import List

import torch.nn as nn

from labml import lab
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.mnist import MNISTConfigs

from .cifar100data import CIFAR100Configs as CIFAR100DatasetConfigs


class CIFAR100Configs(CIFAR100DatasetConfigs, MNISTConfigs):
    """
    ## Configurations

    This extends from CIFAR 10 dataset configurations from
     [`labml_helpers`](https://github.com/labmlai/labml/tree/master/helpers)
     and [`MNISTConfigs`](mnist.html).
    """
    # Use CIFAR100 dataset by default
    dataset_name: str = 'CIFAR100'


@option(CIFAR100Configs.train_dataset)
def cifar100_train_augmented():
    """
    ### Augmented CIFAR 10 train dataset
    """
    from torchvision.datasets import CIFAR100
    from torchvision.transforms import transforms
    return CIFAR100(str(lab.get_data_path()),
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       # Pad and crop
                       transforms.RandomCrop(32, padding=4),
                       # Random horizontal flip
                       transforms.RandomHorizontalFlip(),
                       #
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))


@option(CIFAR100Configs.valid_dataset)
def cifar100_valid_no_augment():
    """
    ### Non-augmented CIFAR 10 validation dataset
    """
    from torchvision.datasets import CIFAR100
    from torchvision.transforms import transforms
    return CIFAR100(str(lab.get_data_path()),
                   train=False,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
