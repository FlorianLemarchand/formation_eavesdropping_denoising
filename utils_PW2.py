"""
Authors: Florian Lemarchand, Maxime Pelcat
Date: 2020
"""

import torch
import random
import numpy as np
import torch.nn as nn
from datetime import datetime
from skimage.io import imread, imsave
from os import listdir, path, makedirs
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from skimage.util import random_noise, img_as_ubyte, img_as_float32
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def make_learning_set():
    # List the files contained in the data directory
    filenames = listdir('data/in/bsd')
    print('Directory contains {} images!'.format(len(filenames)))

    # Shuffle the array of filenames to ensure random distribution into sets
    np.random.seed(0)
    np.random.shuffle(filenames)

    # Separate the filenames in three sets : 80% train 10% val 10% test
    train_ratio = 0.8
    n_train = int(len(filenames) * train_ratio)
    n_val_test = int((len(filenames) - n_train) / 2)

    train_filenames = filenames[0:n_train]
    val_filenames = filenames[n_train: n_train + n_val_test]
    test_filenames = filenames[n_train + n_val_test: n_train + 2 * n_val_test]

    set_list = [train_filenames, val_filenames, test_filenames]

    print('{} train samples, {} val samples, {} test samples'.format(len(train_filenames),
                                                                     len(val_filenames),
                                                                     len(test_filenames)))

    # Create the directories
    for subset in ['train', 'val', 'test']:
        for folder in ['ref', 'in']:
            makedirs('data/out/bsd_learning/{}/{}'.format(subset, folder), exist_ok=True)

    # Loop over subsets and their files
    for i, subset in enumerate(['train', 'val', 'test']):
        for j, f in enumerate(set_list[i]):
            if ((j + 1) % 10 is 0) and ((j + 1) is not 0):
                print('{}: {}/{}'.format(subset, j + 1, len(set_list[i])))
            # Copy and open the reference file
            input_path = path.join('data/in/bsd', f)
            im_ref = imread(input_path, as_gray=True)

            # Noise 'im_ref'
            # TO DO PW2 / Q1
            # var =
            im_noise = 0
            # END TO DO PW2 / Q1

            # Save the image
            name, ext = f.split('.')
            f = name + '.png'  # Change the file extension

            output_path_ref = path.join('data/out/bsd_learning', subset, 'ref', f)
            output_path_in = path.join('data/out/bsd_learning', subset, 'in', f)

            im_ref = img_as_ubyte(im_ref)  # Cast to int for saving

            # If subset is train or val, crop square patches in images. This enhance variability in training and
            # gets homogeneous batches. If test, save the samples as is.
            if subset is not 'test':
                patch_and_save(im_ref, output_path_ref, 128)
                patch_and_save(im_noise, output_path_in, 128)
            else:
                imsave(output_path_ref, im_ref)
                imsave(output_path_in, im_noise)


def make_learning_set_intercept():
    # List the files contained in the data directory
    filenames = listdir('data/in/intercept/in')

    # Shuffle the array of filenames to ensure random distribution into sets
    np.random.seed(0)
    np.random.shuffle(filenames)

    # Separate the filenames in three sets : 80% train 10% val 10% test
    train_ratio = 0.8
    n_train = int(len(filenames) * train_ratio)
    n_val_test = int((len(filenames) - n_train) / 2)

    train_filenames = filenames[0:n_train]
    val_filenames = filenames[n_train: n_train + n_val_test]
    test_filenames = filenames[n_train + n_val_test: n_train + 2 * n_val_test]

    set_list = [train_filenames, val_filenames, test_filenames]

    print('{} train samples, {} val samples, {} test samples'.format(len(train_filenames),
                                                                     len(val_filenames),
                                                                     len(test_filenames)))

    # Create the directories
    for subset in ['train', 'val', 'test']:
        for folder in ['ref', 'in']:
            makedirs('data/out/bsd_learning_intercept/{}/{}'.format(subset, folder), exist_ok=True)

    # Loop over subsets and their files
    for i, subset in enumerate(['train', 'val', 'test']):
        for j, f in enumerate(set_list[i]):
            if ((j + 1) % 10 is 0) and ((j + 1) is not 0):
                print('{}: {}/{}'.format(subset, j + 1, len(set_list[i])))
            # Copy and open the reference and input files
            reference_path = path.join('data/in/intercept/ref', f)
            im_ref = imread(reference_path, as_gray=True)

            input_path = path.join('data/in/intercept/in', f)
            im_noise = imread(input_path, as_gray=True)

            # Save the image
            name, ext = f.split('.')
            f = name + '.png'  # Change the file extension

            output_path_ref = path.join('data/out/bsd_learning_intercept', subset, 'ref', f)
            output_path_in = path.join('data/out/bsd_learning_intercept', subset, 'in', f)

            # If subset is train or val, crop square patches in images. This enhance variability in training and gets
            # homogeneous batches (square images instead of vertical and horizontal). If test, save the samples as is.
            if subset is not 'test':
                patch_and_save(im_ref, output_path_ref, 128)
                patch_and_save(im_noise, output_path_in, 128)
            else:
                imsave(output_path_ref, im_ref)
                imsave(output_path_in, im_noise)


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


class REDNet30(nn.Module):
    def __init__(self, num_layers=15, num_features=64):
        super(REDNet30, self).__init__()
        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, 1, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_layers(x)
        out = self.deconv_layers(out)
        out += residual
        out = self.relu(out)
        return out


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, is_test=False):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_paths = listdir(input_dir)
        self.is_test = is_test

    def __getitem__(self, index):
        x = img_as_float32(imread(path.join(self.input_dir, self.input_paths[index]), as_gray=True))
        y = img_as_float32(imread(path.join(self.target_dir, self.input_paths[index]), as_gray=True))

        if not self.is_test:
            x, y = data_augmentation(x, y)

        # Add channel dimension
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        # Convert numpy array to torch tensor
        x = torch.tensor(x.copy())  # required copy: memory arrangement of numpy striding not supported by pytorch
        y = torch.tensor(y.copy())

        # x = torch.from_numpy(x.copy())
        # y = torch.from_numpy(y.copy())

        return x, y

    def __len__(self):
        return len(self.input_paths)


def data_augmentation(input, target):
    # Random choice of the augmentation components
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    # TO DO PW2 / Q2
    # if hflip:
    #
    # if vflip:
    #
    # if rot90:
    # END TO DO PW2 / Q2

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


def ensemble_inference(model, im_tens):
    # List of transforms to apply on the input images
    transforms = [im_tens,
                  im_tens.transpose(-2, -1).flip(-1),
                  im_tens.transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1),
                  im_tens.transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1),
                  im_tens.flip(-1),
                  im_tens.flip(-1).transpose(-2, -1).flip(-1),
                  im_tens.flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1),
                  im_tens.flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1)]

    # Pass the transformed images through the model
    out = []
    for tens in transforms:
        out.append(model(tens)[0])

    # Transforms back the results so they can be fused
    transforms_back = torch.cat(
        (out[0].unsqueeze(dim=0),
         out[1].flip(-1).transpose(-2, -1).unsqueeze(dim=0),
         out[2].flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).unsqueeze(dim=0),
         out[3].flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).unsqueeze(dim=0),
         out[4].flip(-1).unsqueeze(dim=0),
         out[5].flip(-1).transpose(-2, -1).flip(-1).unsqueeze(dim=0),
         out[6].flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1).unsqueeze(dim=0),
         out[7].flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1).flip(-1).unsqueeze(
             dim=0)
         ), dim=0)

    # Get the mean of the 8 images obtained by passing the transformed image through the model
    ensemble_out = torch.mean(transforms_back, dim=0)

    # Remove useless dimension
    ensemble_out_uns = ensemble_out.unsqueeze(dim=0)
    return ensemble_out_uns


def question_solver(question_number, part):
    if (part is 1) and (question_number in [1, 2, 3, 4]):
        return True
    elif (part is 2) and (question_number in [2, 3, 4]):
        return True
    elif (part is 3) and (question_number in [3, 4]):
        return True
    elif (part is 4) and (question_number in [4]):
        return True
    elif (part is 5) and (question_number in [5]):
        return True
    elif (part is 6) and (question_number in [6]):
        return True
    elif (part is 7) and (question_number in [7]):
        return True
    elif (part is 8) and (question_number in [8]):
        return True
    else:
        return False
