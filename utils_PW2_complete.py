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
from datetime import datetime
from skimage.io import imread, imsave
from os import listdir, path, makedirs
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from skimage.util import random_noise, img_as_ubyte, img_as_float32


class DnCNN(nn.Module):
    """
    DnCNN code from https://github.com/SaoYan/DnCNN-PyTorch
    """

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

    set_list = [train_filenames, val_filenames, test_filenames]

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
        for j, f in enumerate(set_list[i]):
            if (j % 10 is 0) and (j is not 0):
                print('Subset {} : {}/{}'.format(subset, j, len(set_list[i])))
            # Copy the reference file
            input_path = path.join('data/in/bsd', f)
            im_ref = imread(input_path, as_gray=True)
            ####### TO DO PW2  #######
            # Noise 'im_ref'
            var = np.square(50./255.)
            im_noise = img_as_ubyte(random_noise(im_ref, 'gaussian', mean=0., var=var,  clip=True))
            # Save the image
            ####### END TO DO PW2 #######

            output_path_ref = path.join('data/out/bsd_learning', subset, 'ref', f)
            output_path_in = path.join('data/out/bsd_learning', subset, 'in', f)

            im_ref = img_as_ubyte(im_ref)

            if subset is not 'test':
                patch_and_save(im_ref, output_path_ref, 64)
                patch_and_save(im_noise, output_path_in, 64)

            else:

                imsave(output_path_ref, im_ref)
                imsave(output_path_in, im_noise)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_paths = listdir(input_dir)

    def __getitem__(self, index):
        x = img_as_float32(imread(path.join(self.input_dir, self.input_paths[index]), as_gray=True))
        y = img_as_float32(imread(path.join(self.target_dir, self.input_paths[index]), as_gray=True))

        x, y = data_augmentation(x, y)

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        x = torch.from_numpy(x.copy())
        y = torch.from_numpy(y.copy())

        return x, y

    def __len__(self):
        return len(self.input_paths)


def data_augmentation(input, target):
    # Random choice of the augmentation components
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    # TO DO PW2
    if hflip:
        input = input[:, ::-1]
        target = target[:, ::-1]
    if vflip:
        input = input[::-1, :]
        target = target[::-1, :]
    if rot90:
        input = input.transpose(1, 0)
        target = target.transpose(1, 0)

    return input, target


def patch_and_save(img, save_dir, patch_size):
    patches = []
    h, w = img.shape
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patches.append(img[i: i + patch_size, j: j + patch_size])

    base, filename = path.split(save_dir)
    name, ext = filename.split('.')

    for i, p in enumerate(patches):
        save_path = path.join(base, '{}_{}.{}'.format(name, i, ext))
        # print(save_path)
        imsave(save_path, p, check_contrast=False)


def generate_logdir(root_path):
    # Make the log_dir if it does not exist
    makedirs(root_path, exist_ok=True)
    # Create an unique name based on the creation date
    date = datetime.today()
    log_name = str(date.day) + "_" + str(date.month) + "_" + str(date.year) + "_" + \
               str(date.hour) + "_" + str(date.minute) + "_" + str(date.second)
    logdir_name = path.join(root_path, log_name)
    makedirs(logdir_name)
    return logdir_name


class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        # break

    def add_scalar(self, index, val, niter):
        self.writer.add_scalar(index, val, niter)

    def add_scalars(self, index, group_dict, niter):
        self.writer.add_scalar(index, group_dict, niter)

    def add_image_grid(self, index, ngrid, x, niter):
        grid = make_grid(x, ngrid)
        self.writer.add_image(index, grid, niter)

    def add_image_single(self, index, x, niter):
        self.writer.add_image(index, x, niter)

    def add_graph(self, x_input, model):
        self.writer.add_graph(model, x_input)

    def export_json(self, out_file):
        self.writer.export_scalars_to_json(out_file)