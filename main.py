import os
import sys
import argparse
import logging
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.transforms as transforms

from vocData import vocData
from utils.transform import MaskToTensor
import models
from utils.functions import get_criterion, train, test


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])

def parser_args():
    parser = argparse.ArgumentParser(description='Train Mnist dataset')
    parser.add_argument('--epoch', help='Total epoches', default=50, type=int)
    parser.add_argument('--batch_size', help='Batch size', default=1, type=int)
    parser.add_argument('--loss', help='The loss function', default='crossentropy', type=str)
    parser.add_argument('--lr', help='The learning rate', default=0.001, type=float)
    parser.add_argument('--ignore_label', help='The index of label for ignore', default=255, type=int)
    parser.add_argument('--root', help='The initial dataset root', default='./VOCdevkit/VOC2012', type=str)
    parser.add_argument('--weight', help='The weight folder', default='./weights', type=str)
    parser.add_argument('--model', help='The architecture of CNN', default='FCN8s', type=str)
    parser.add_argument('--workers', help='Number of workers when loading data', default=4, type=int)
    parser.add_argument('--print_freq', help='Number of iterations to print', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    args = parser.parse_args()
    return args

def main(net, dataloader, device, config):
    train_loader, test_loader = dataloader[0], dataloader[1]
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    crit = get_criterion(config, ignore_label=config.ignore_label)
    crit = crit.to(device)

    if not os.path.isdir(config.weight):
        os.makedirs(config.weight)
    checkpoint = os.path.join(config.weight, config.model)

    for epoch in range(config.epoch):
        ########### TRAIN ##########
        start = time()
        loss_train = train(config, net, device, train_loader, crit, optimizer, epoch)
        end = time() - start
        logging.info(
            f'=> Epoch[{epoch}] finished, Average train Loss: {loss_train:.3f}, Tot Time: {end:.3f}'
        )
        ########### TEST ###########
        start = time()
        PA, mPA, mIoU = test(config, net, device, test_loader, epoch)
        end = time() - start
        logging.info(
            f'=> Epoch[{epoch}] Test Result'
        )
        logging.info(
            f'=> Pixel Accuracy: {PA:.3f} | Mean Pixel Accuracy: {mPA:.3f} | Mean IoU: {mIoU:.3f}'
        )
        torch.save(net.state_dict(), checkpoint + '_%d.pth'%epoch)


if __name__ == '__main__':
    config = parser_args()
    logging.info(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = eval('models.' + config.model + '.get_CNN')(config)
    net.to(device)

    # transform of images
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    mask_transform = MaskToTensor()

    train_dataset = vocData(
        config.root, 
        'train', 
        transform=input_transform, 
        mask_transform=mask_transform)
    train_loader = Data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.workers, 
        pin_memory=True)

    test_dataset = vocData(
        config.root, 
        'val', 
        transform=input_transform, 
        mask_transform=mask_transform)
    test_loader = Data.DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.workers, 
        pin_memory=True)

    loader = (train_loader, test_loader)
    main(net, loader, device, config)
