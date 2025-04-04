from .distributed_dataset import DistributedDataset, distributed_dataset
from .cifar100 import cifar100
from .cifar10 import cifar10
from .tiny_imagenet import tiny_imagenet
from .mnist import mnist


def get_dataset(rank, dataset_name,
                split=None, batch_size=None,
                transforms=None, is_distribute=True,
                seed=777, path="../data", **kwargs):
    if dataset_name == "CIFAR10":
        return cifar10(rank=rank,
                       split=split,
                       batch_size=batch_size,
                       transforms=transforms,
                       is_distribute=is_distribute,
                       seed=seed,
                       path=path,
                       **kwargs)
    elif dataset_name == "CIFAR100":
        return cifar100(rank=rank,
                        split=split,
                        batch_size=batch_size,
                        transforms=transforms,
                        is_distribute=is_distribute,
                        seed=seed,
                        path=path,
                        **kwargs)
    elif dataset_name == "TinyImageNet":
        return tiny_imagenet(rank=rank,
                             split=split,
                             batch_size=batch_size,
                             transforms=transforms,
                             is_distribute=is_distribute,
                             seed=seed,
                             path=path,
                             **kwargs)
    elif dataset_name == "MNIST":
        return mnist(rank=rank,
                     split=split,
                     batch_size=batch_size,
                     transforms=transforms,
                     is_distribute=is_distribute,
                     seed=seed,
                     path=path,
                     **kwargs)