from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from labml import lab
from labml.configs import BaseConfigs


def _dataset(is_train, transform):
    return datasets.CIFAR100(str(lab.get_data_path()),
                            train=is_train,
                            download=True,
                            transform=transform)


class CIFAR100Configs(BaseConfigs):
    """
    Configurable CIFAR 10 data set.

    Arguments:
        dataset_name (str): name of the data set, ``CIFAR100``
        dataset_transforms (torchvision.transforms.Compose): image transformations
        train_dataset (torchvision.datasets.CIFAR100): training dataset
        valid_dataset (torchvision.datasets.CIFAR100): validation dataset

        train_loader (torch.utils.data.DataLoader): training data loader
        valid_loader (torch.utils.data.DataLoader): validation data loader

        train_batch_size (int): training batch size
        valid_batch_size (int): validation batch size

        train_loader_shuffle (bool): whether to shuffle training data
        valid_loader_shuffle (bool): whether to shuffle validation data
    """
    dataset_name: str = 'CIFAR100'
    dataset_transforms: transforms.Compose
    train_dataset: datasets.CIFAR100
    valid_dataset: datasets.CIFAR100

    train_loader: DataLoader
    valid_loader: DataLoader

    train_batch_size: int = 64
    valid_batch_size: int = 1024

    train_loader_shuffle: bool = True
    valid_loader_shuffle: bool = False


@CIFAR100Configs.calc(CIFAR100Configs.dataset_transforms)
def cifar100_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


@CIFAR100Configs.calc(CIFAR100Configs.train_dataset)
def cifar100_train_dataset(c: CIFAR100Configs):
    return _dataset(True, c.dataset_transforms)


@CIFAR100Configs.calc(CIFAR100Configs.valid_dataset)
def cifar100_valid_dataset(c: CIFAR100Configs):
    return _dataset(False, c.dataset_transforms)


@CIFAR100Configs.calc(CIFAR100Configs.train_loader)
def cifar100_train_loader(c: CIFAR100Configs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=c.train_loader_shuffle)


@CIFAR100Configs.calc(CIFAR100Configs.valid_loader)
def cifar100_valid_loader(c: CIFAR100Configs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=c.valid_loader_shuffle)


CIFAR100Configs.aggregate(CIFAR100Configs.dataset_name, 'CIFAR100',
                       (CIFAR100Configs.dataset_transforms, 'cifar100_transforms'),
                       (CIFAR100Configs.train_dataset, 'cifar100_train_dataset'),
                       (CIFAR100Configs.valid_dataset, 'cifar100_valid_dataset'),
                       (CIFAR100Configs.train_loader, 'cifar100_train_loader'),
                       (CIFAR100Configs.valid_loader, 'cifar100_valid_loader'))
