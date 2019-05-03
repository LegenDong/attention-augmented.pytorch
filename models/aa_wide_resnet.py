# -*- coding: utf-8 -*-
# @Time    : 2019/5/3 14:01
# @Author  : LegenDong
# @User    : legendong
# @File    : aa_wide_resnet.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.augmented_layer import AugmentedLayer

__all__ = ['AAWideResNet']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class AAWideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, k=0.2, v=0.1, Nh=8, fh=32, fw=32):
        super(AAWideBasic, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = AugmentedLayer(in_planes, planes, kernel_size=3, padding=1,
                                    dk=int(k * planes), dv=int(v * planes), Nh=Nh, relative=True, fh=fh, fw=fw)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class AAWideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, aa_k=0.2, aa_v=0.1, Nh=8, fh=32, fw=32):
        super(AAWideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'AA-Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        print('| AA-Wide-Resnet %dx%d' % (depth, k))
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, n_stages[0])
        self.layer1 = self._wide_layer(AAWideBasic, n_stages[1], n, dropout_rate, stride=1,
                                       aa_k=aa_k, aa_v=aa_v, Nh=Nh, fh=fh, fw=fw)
        self.layer2 = self._wide_layer(AAWideBasic, n_stages[2], n, dropout_rate, stride=2,
                                       aa_k=aa_k, aa_v=aa_v, Nh=Nh, fh=fh, fw=fw)
        self.layer3 = self._wide_layer(AAWideBasic, n_stages[3], n, dropout_rate, stride=2,
                                       aa_k=aa_k, aa_v=aa_v, Nh=Nh, fh=fh // 2, fw=fw // 2)
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.linear = nn.Linear(n_stages[3], num_classes)

        # not sure about the init func for the AA-Wide-ResNet, maybe this will work
        for m in self.modules():
            conv_init(m)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, aa_k, aa_v, Nh, fh, fw):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        last_stride = 1
        for stride in strides:
            fh //= last_stride
            fw //= last_stride
            last_stride = stride
            layers.append(block(self.in_planes, planes, dropout_rate, stride, aa_k, aa_v, Nh, fh, fw))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    model = AAWideResNet(28, 10, 0.3, 10)
    img_data = torch.randn(1, 3, 32, 32)
    y = model(img_data)
    print(y.size())
