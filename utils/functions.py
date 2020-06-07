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
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

from utils.evaluate import pixel_accuracy, mean_pixel_accuracy, mean_IOU, torch2np, fw_IOU


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

def validate(config, net, device, test_loader):
    logging.info(
        f'Start validating...'
    )
    net.eval()
    PA = 0
    mPA = 0
    mIoU = 0
    fwIoU = 0
    size = len(test_loader)
    for i, (img, label) in enumerate(test_loader):
        img = img.to(device)
        label = label.to(device)
        output = net(img)
        
        curr_PA = pixel_accuracy(output, label)
        curr_mPA = mean_pixel_accuracy(output, label)
        curr_mIoU = mean_IOU(output, label)
        curr_fwIoU = fw_IOU(output, label)

        PA += curr_PA
        mPA += curr_mPA
        mIoU += curr_mIoU
        fwIoU += curr_fwIoU

        if (i + 1) % config.print_freq == 0:
            logging.info(
                f'Testing[{i + 1}/{size}], PA: {curr_PA:.3f}({PA/(i+1):.3f}), mPA: {curr_mPA:.3f}({mPA/(i+1):.3f}), mIoU: {curr_mIoU:.3f}({mIoU/(i+1):.3f}), fw_IoU: {curr_fwIoU:.3f}({fwIoU/(i+1):.3f})'
            )
    
    PA, mPA, mIoU, fwIoU = PA / size, mPA / size, mIoU / size, fwIoU / size

    logging.info(
            f'=> Pixel Accuracy: {PA:.3f} | Mean Pixel Accuracy: {mPA:.3f} | Mean IoU: {mIoU:.3f} | Freq Weight IoU: {fwIoU:.3f}\n'
        )
    
    return PA, mPA, mIoU, fwIoU

def visualize(config, preds, masks, idx):
    palette=[]
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21]=np.array([[0, 0, 0],
                                [128, 0, 0],
                                [0, 128, 0],
                                [128, 128, 0],
                                [0, 0, 128],
                                [128, 0, 128],
                                [0, 128, 128],
                                [128, 128, 128],
                                [64, 0, 0],
                                [192, 0, 0],
                                [64, 128, 0],
                                [192, 128, 0],
                                [64, 0, 128],
                                [192, 0, 128],
                                [64, 128, 128],
                                [192, 128, 128],
                                [0, 64, 0],
                                [128, 64, 0],
                                [0, 192, 0],
                                [128, 192, 0],
                                [0, 64, 128]
                                ], dtype='uint8').flatten()

    preds, masks = torch2np(preds, masks)
    pred = preds[0, :, :].astype(np.uint8)
    mask = masks[0, :, :].astype(np.uint8)

    pred = Image.fromarray(pred, mode='P')
    pred.putpalette(palette)
    mask = Image.fromarray(mask, mode='P')
    mask.putpalette(palette)

    if not os.path.isdir(config.image_root):
        os.makedirs(config.image_root)

    pred_name = os.path.join(config.image_root, config.model + '_%d_pred.png'%idx)
    mask_name = os.path.join(config.image_root, config.model + '_%d_mask.png'%idx)
    logging.info(f'Saving {pred_name} and {mask_name}.')
    pred.save(pred_name)
    mask.save(mask_name)
