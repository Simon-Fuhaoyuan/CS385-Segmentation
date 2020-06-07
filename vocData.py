import copy
import logging
import random
import sys

from PIL import Image
import scipy.io as sio
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.transform import MaskToTensor


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])
logger = logging.getLogger(__name__)

class vocData(Dataset):
    def __init__(self, root, phase, transform=None, mask_transform=None):
        assert phase in ['train', 'val'], 'Unknown phase for %s'%phase

        self.root = root
        self.ignore_label = 255
        self.phase = phase
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = self.make_dataset()
        logger.info('=> Loading {} images from {}'.format(self.phase, self.root))
        logger.info('=> num_images: {}'.format(len(self.imgs)))
    
    def make_dataset(self):
        items = []

        if self.root[-4:] == '2012': # VOC2012
            img_root = os.path.join(self.root, 'JPEGImages')
            mask_root = os.path.join(self.root, 'SegmentationClass')
            data_list = [l.strip('\n') for l in open(os.path.join(
                self.root, 'ImageSets', 'Segmentation', self.phase + '.txt')).readlines()]
            for it in data_list:
                item = (os.path.join(img_root, it + '.jpg'), os.path.join(mask_root, it + '.png'))
                items.append(item)
        else: # VOC SBD
            img_root = os.path.join(self.root, 'img')
            mask_root = os.path.join(self.root, 'SegmentationClass')
            data_list = [l.strip('\n') for l in open(os.path.join(
                self.root, self.phase + '.txt')).readlines()]
            for it in data_list:
                item = (os.path.join(img_root, it + '.jpg'), os.path.join(mask_root, it + '.png'))
                items.append(item)

        return items

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask
    
    def __len__(self):
        return len(self.imgs)

    
if __name__ == "__main__":
    root = 'VOCdevkit/VOC2012'
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    mask_transform = MaskToTensor()
    # target_transform = extended_transforms.MaskToTensor()
    train_set = vocData(root, 'train', transform=input_transform, mask_transform=mask_transform)
    train_loader = DataLoader(train_set, batch_size=1, num_workers=4, shuffle=True)
    for i, (img, mask) in enumerate(train_loader):
        print(img.shape)
        # print(mask)
        labels = np.zeros(256)
        for i in range(256):
            labels[i] = (mask == i).sum()
        print(labels)
        exit()
