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

from MnistDataLoader import Mnist
from evaluate import accuracy
import models
from MnistData import load_mnist

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])


def get_criterion(config):
    if config.loss == 'crossentropy':
        crit = nn.CrossEntropyLoss()
    else:
        logging.info(f'Error: No match criterion')
        exit()

    return crit