import os
import sys
import argparse
import logging
from time import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.transforms as transforms

from vocData import vocData
from utils.transform import MaskToTensor
from utils.functions import validate
import models


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])

def parser_args():
    parser = argparse.ArgumentParser(description='Valid Pascal VOC dataset')
    parser.add_argument('--batch_size', help='Batch size', default=1, type=int)
    parser.add_argument('--root', help='The initial dataset root', default='./VOCdevkit/VOC2012', type=str)
    parser.add_argument('--model', help='The architecture of CNN', default='FCN8s', type=str)
    parser.add_argument('--workers', help='Number of workers when loading data', default=4, type=int)
    parser.add_argument('--checkpoint', help='The model weight', default='weights/FCN8s_best.pth', type=str)
    parser.add_argument('--print_freq', help='Number of iterations to print', default=200, type=int)

    args = parser.parse_args()
    return args

def main(net, test_loader, device, config):
    validate(config, net, device, test_loader)


if __name__ == '__main__':
    config = parser_args()
    logging.info(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = eval('models.' + config.model + '.get_CNN')(config)
    net.to(device)
    checkpoint = config.checkpoint
    logging.info(f'Loading weight parameters from {checkpoint}.')
    net.load_state_dict(torch.load(checkpoint))

    # transform of images
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    mask_transform = MaskToTensor()

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

    main(net, test_loader, device, config)
