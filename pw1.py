"""
Authors: Florian Lemarchand, Maxime Pelcat
Date: 2020
"""

# ========= Imports =========
import numpy as np
from os import makedirs
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from matplotlib.colors import LogNorm
from skimage.io import imread, imsave
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from skimage.util import random_noise, img_as_ubyte, img_as_float
from scipy.ndimage import gaussian_filter, median_filter, convolve
from skimage.metrics import peak_signal_noise_ratio, \
    mean_squared_error, \
    structural_similarity


# ========= Utils =========
def print_psnr_ssim(im1, im2, label):
    psnr = np.round(peak_signal_noise_ratio(im1, im2), 2)
    ssim = np.round(structural_similarity(im1, im2), 2)
    print('{}: PSNR: {} / SSIM: {}'.format(label, psnr, ssim))


def print_line_break():
    print('\n')

# ========= Write your code hereunder =========
