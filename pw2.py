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

from pybm3d.bm3d import bm3d


# ========= Utils =========
def print_psnr_ssim(im1, im2, label):
    psnr = np.round(peak_signal_noise_ratio(im1, im2), 2)
    ssim = np.round(structural_similarity(im1, im2), 2)
    print('{}: PSNR: {} / SSIM: {}'.format(label, psnr, ssim))


def print_line_break():
    print('\n')

# ========= Write your code hereunder =========
# im1 = img_as_ubyte(imread('data/in/intercept1.bmp', as_gray=True))
im1 = img_as_ubyte(imread('data/in/im1.jpg', as_gray=True))

im1_gray_int = rgb2gray(im1)

sigma = 50
# Cast the image and parameters to float : each value coded on 64-bit float with value in [0,1]
im1_gray = img_as_float(im1_gray_int)
sigma_float = sigma/255.
variance = np.square(sigma_float)  # variance = np.square(std)


im_noise_lib = img_as_ubyte(random_noise(im1_gray, 'gaussian', mean=0., seed=0, var=variance, clip=True))
# im_noise_lib = img_as_ubyte(random_noise(im1_gray, 's&p', amount=0.5))

im_denoised = bm3d(im_noise_lib, 50)


plt.subplot(2,2,1)
plt.imshow(im1_gray, cmap='gray')
plt.subplot(2,2,2)
plt.imshow(im_noise_lib, cmap='gray')
plt.subplot(2,2,3)
plt.imshow(im_denoised, cmap='gray')
plt.show()

print_psnr_ssim(im_noise_lib, im1_gray_int, 'noisy')

print_psnr_ssim(im_denoised, im1_gray_int, 'denoised1')


# plt.show()