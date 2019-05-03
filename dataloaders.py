# -*- coding: utf-8 -*-
# @Time    : 2019/5/3 16:43
# @Author  : LegenDong
# @User    : legendong
# @File    : dataloaders.py
# @Software: PyCharm
import torchvision
from torch.utils import data
from torchvision import transforms

__all__ = ['Cifar100DataLoader']


class Cifar100DataLoader(data.DataLoader):
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)

    def __init__(self, data_root, batch_size, shuffle, num_workers, training=True):
        if training:
            trsfm = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])

        dataset = torchvision.datasets.CIFAR100(data_root, train=training, download=True, transform=trsfm)

        self.n_samples = len(dataset)

        super(Cifar100DataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers)
