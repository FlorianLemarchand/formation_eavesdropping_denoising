"""
Authors: Florian Lemarchand, Maxime Pelcat
Date: 2020
"""

import sys
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from os import listdir, path

'''
DnCNN code from https://github.com/SaoYan/DnCNN-PyTorch

'''


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, transforms=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_paths = listdir(input_dir)
        self.transforms = transforms

    def __getitem__(self, index):
        # print(path.join(input_dir, self.input_paths[index]))
        x = Image.open(path.join(self.input_dir, self.input_paths[index]))
        # print(path.join(target_dir, (self.input_paths[index])[0:-6]+'.png'))
        y = Image.open(path.join(self.target_dir, (self.input_paths[index])[0:-6] + '_gt.png'))

        seed = np.random.randint(sys.maxsize)
        if self.transforms:
            random.seed(seed)  # Ensure the same tranformations for input and target
            x = self.transforms(x)
            random.seed(seed)  # Ensure the same tranformations for input and target
            y = self.transforms(y)
        return x, y

    def __len__(self):
        return len(self.input_paths)
