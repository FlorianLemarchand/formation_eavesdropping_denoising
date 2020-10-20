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
from shutil import copyfile
from os import listdir, path, makedirs

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


def make_learning_set():
    # List the files contained in the data directory
    filenames = listdir('data/in/bsd')
    print('Directory contains {} images!'.format(len(filenames)))

    # Shuffle the array of filenames to ensure random distribution into sets
    np.random.shuffle(filenames)

    # Separate in three sets
    train_ratio = 0.8
    n_train = int(len(filenames) * train_ratio)
    n_val = int((len(filenames) - n_train) / 2)

    train_filenames = filenames[0:n_train]
    val_filenames = filenames[n_train: n_train + n_val]
    test_filenames = filenames[n_train + n_val: n_train + 2 * n_val]

    print('{} train samples, {} train samples, {} train samples'.format(len(train_filenames),
                                                                        len(val_filenames),
                                                                        len(test_filenames)))

    # Create the directories
    makedirs('data/out/bsd_learning/train/ref', exist_ok=True)
    makedirs('data/out/bsd_learning/train/in', exist_ok=True)

    makedirs('data/out/bsd_learning/val/ref', exist_ok=True)
    makedirs('data/out/bsd_learning/val/in', exist_ok=True)

    makedirs('data/out/bsd_learning/test/ref', exist_ok=True)
    makedirs('data/out/bsd_learning/test/in', exist_ok=True)

    # Loop over subsets and files
    for i, subset in enumerate(['train', 'val', 'test']):
        for f in [train_filenames, val_filenames, test_filenames][i]:
            # Copy the reference file
            input_path = path.join('data/in/bsd', f)
            output_path = path.join('data/out/bsd_learning', subset, 'ref', f)
            copyfile(input_path, output_path)
            # Read, noise and save the noisy version
            ####### TO DO PW2  #######
            # Read the image

            # Noise the image

            # Save the image

            ####### END TO DO PW2 #######


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_paths = listdir(input_dir)

    def __getitem__(self, index):
        # print(path.join(input_dir, self.input_paths[index]))
        x = Image.open(path.join(self.input_dir, self.input_paths[index]))
        # print(path.join(target_dir, (self.input_paths[index])[0:-6]+'.png'))
        y = Image.open(path.join(self.target_dir, self.input_paths[index]))

        x, y = data_augmentation(x, y)
        return x, y

    def __len__(self):
        return len(self.input_paths)

def data_augmentation(input, target):
    # Random choice of the augmentation components
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    # TO DO PW2
    # if hflip:
    #     input =
    #     target =
    # if vflip:
    #     input =
    #     target =
    # if rot90:
    #     input =
    #     target =

    return input, target
