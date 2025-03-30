import os

from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from torchvision.datasets import MNIST
from .distributed_dataset import distributed_dataset


def mnist(rank, split=None, batch_size=None,
          transforms=None, test_batch_size=64,
          is_distribute=True, seed=777, path="../data", **kwargs):
    if transforms is None:
        transforms = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize((0.1307,), (0.3081,))
        ])
    if batch_size is None:
        batch_size = 1
    if split is None:
        split = [1.0]
    if not os.path.exists(path):
        os.mkdir(path)
    train_set = MNIST(root=path, train=True, download=True, transform=transforms)
    test_set = MNIST(root=path, train=False, download=True, transform=transforms)
    if is_distribute:
        train_set = distributed_dataset(train_set, split, rank, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, drop_last=True)
    return train_loader, test_loader, (1, 28, 28), 10