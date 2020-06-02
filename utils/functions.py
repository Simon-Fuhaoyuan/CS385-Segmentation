import os
import sys
import argparse
import logging
import numpy as np
from time import time
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from sklearn import manifold

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

from utils.evaluate import pixel_accuracy, mean_pixel_accuracy, mean_IOU


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])


def get_criterion(config, ignore_label=None):
    if config.loss == 'crossentropy':
        if ignore_label is not None:
            crit = nn.CrossEntropyLoss(ignore_index=ignore_label)
        else:
            crit = nn.CrossEntropyLoss()
    else:
        logging.info(f'Error: No match criterion')
        exit()

    return crit

def train(config, net, device, train_loader, crit, optimizer, epoch):
    net.train()
    loss_train = 0.0
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        output = net(img)
        loss = crit(output, label)
        loss_train += loss.item()

        if i % config.print_freq == 0:
            logging.info(
                f'Epoch[{epoch}][{i}/{len(train_loader)}], Train Loss: {loss.item():.3f}({loss_train / (i + 1):.3f})'
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train /= len(train_loader)
    return loss_train

def test(config, net, device, test_loader, epoch):
    net.eval()
    PA = 0
    mPA = 0
    mIoU = 0
    size = len(test_loader)
    for i, (img, label) in enumerate(test_loader):
        img = img.to(device)
        label = label.to(device)
        output = net(img)
        
        PA += pixel_accuracy(output, label)
        mPA += mean_pixel_accuracy(output, label)
        mIoU += mean_IOU(output, label)

        if i % config.print_freq == 0:
            logging.info(
                f'Testing Epoch[{epoch}][{i}/{len(test_loader)}], PA: {PA/(i+1):.3f}, mPA: {mPA/(i+1):.3f}, mIoU: {mIoU/(i+1):.3f}'
            )
    
    return PA / size, mPA / size, mIoU / size