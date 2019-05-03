# -*- coding: utf-8 -*-
# @Time    : 2019/5/3 14:03
# @Author  : LegenDong
# @User    : legendong
# @File    : main.py
# @Software: PyCharm
import argparse
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from dataloaders import Cifar100DataLoader
from models.aa_wide_resnet import AAWideResNet


def check_exists(file_paths):
    if not isinstance(file_paths, (list, tuple)):
        file_paths = [file_paths]
    for file_path in file_paths:
        if not os.path.exists(file_path):
            return False
    return True


def save_model(model, save_path, name):
    save_name = os.path.join(save_path, '{}_best.pth'.format(name))
    torch.save(model.state_dict(), save_name)
    return save_name


def prepare_device():
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    list_ids = list(range(n_gpu))
    return device, list_ids


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def rank1_func(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top1_func(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def topk_func(output, target, k=5):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def run_train(epoch_idx, model, train_loader, optimizer, loss_func, device, log_step):
    model.train()

    total_loss = .0
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        local_loss = loss_func(outputs, target)

        local_loss.backward()
        optimizer.step()

        total_loss += local_loss.item()

        if batch_idx % log_step == 0 and batch_idx != 0:
            end = time.time()

            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Time: {:.2f}'
                  .format(epoch_idx, batch_idx * args.batch_size, train_loader.n_samples,
                          100.0 * batch_idx / len(train_loader), local_loss.item(),
                          end - start))

            start = time.time()

    train_log = {'epoch': epoch_idx,
                 'lr': optimizer.param_groups[0]['lr'],
                 'loss': total_loss / len(train_loader)}

    return train_log


def run_test(model, test_loader, device):
    model.eval()
    total_top1 = .0
    total_top5 = .0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_top1 += top1_func(output, target)
            total_top5 += topk_func(output, target, 5)

    test_log = {'top1 acc': 100. * total_top1 / len(test_loader),
                'top5 acc': 100. * total_top5 / len(test_loader), }

    return test_log


def main(args):
    if not check_exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_loader = Cifar100DataLoader(args.data_root, args.batch_size, shuffle=True, num_workers=4, training=True)
    test_loader = Cifar100DataLoader(args.data_root, args.batch_size * 3, shuffle=True, num_workers=4, training=False)

    train_log_step = len(train_loader) // 100 if len(train_loader) > 100 else 1

    model = AAWideResNet(args.depth, args.widen_factor, args.dropout, args.num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)

    loss_func = torch.nn.CrossEntropyLoss()
    device, device_ids = prepare_device()
    model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    max_test_acc = .0
    for epoch_idx in range(200):
        lr_scheduler.step()

        train_log = run_train(epoch_idx, model, train_loader, optimizer, loss_func, device, train_log_step)
        for key, value in sorted(train_log.items(), key=lambda item: item[0]):
            print('    {:20s}: {:6f}'.format(str(key), value))

        test_log = run_test(model, test_loader, device)
        for key, value in sorted(test_log.items(), key=lambda item: item[0]):
            print('    {:20s}: {:6f}'.format(str(key), value))

        test_acc = test_log['top1 acc']
        if max_test_acc < test_acc:
            state = {
                'model': model.module.state_dict() if len(device_ids) > 1 else model.state_dict(),
                'acc': test_acc,
                'epoch': epoch_idx,
            }
            torch.save(state, os.path.join(args.save_dir, 'best_model.pth'))
            max_test_acc = test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--data_root', default='./data', type=str,
                        help='path to load data (default: ./data)')
    parser.add_argument('--save_dir', default='./checkpoints', type=str,
                        help='path to save model (default: ./checkpoints)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes (default: 100)')
    parser.add_argument('--batch_size', default=64, type=int, help='dim of feature (default: 4096)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help="learning rate for model (default: 0.1)")
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    main(args)
