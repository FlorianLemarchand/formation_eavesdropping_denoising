"""
Authors: Florian Lemarchand, Maxime Pelcat
Date: 2020
"""

# ========= Imports =========
import numpy as np
from os import makedirs
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio, \
    mean_squared_error, \
    structural_similarity

from skimage.util import random_noise, img_as_ubyte, img_as_float
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter, median_filter, convolve

from matplotlib.colors import LogNorm


# ========= Utils =========
def print_psnr_ssim(im1, im2, label):
    psnr = np.round(peak_signal_noise_ratio(im1, im2), 2)
    ssim = np.round(structural_similarity(im1, im2), 2)
    print('{}: PSNR: {} / SSIM: {}'.format(label, psnr, ssim))


# ========= Write your code hereunder =========
# Load image
im1 = imread('data/in/im1.jpg')

# Transform to grayscale
im1_gray = rgb2gray(im1)

sigma = 50
# Cast the image and parameters to float
im1_gray = img_as_float(im1_gray)
sigma_float = sigma / 255.
variance = np.square(sigma_float)  # variance = np.square(std)

# Add AWGN sigma 50 to im1
im_noise_lib = random_noise(im1_gray, 'gaussian', mean=0., seed=0, var=variance, clip=True)
print_psnr_ssim(im_noise_lib, im1_gray, 'Noisy')

# Transform the image and shift the result
fft_clean = fftshift(fft2(im1_gray))
fft_noisy = fftshift(fft2(im_noise_lib))

shape = fft_noisy.shape
middle_y, middle_x = int((shape[0] + 1.)/2.), int((shape[1]+1.)/2.)


# Keep only a part of the coefficients
percentage_keep = 0.2
fft_thresh = fft_noisy.copy()  # init an output image
fft_thresh[:, :] = 0.  # Set to 0. +0.j
half_y_keep = int(((int(percentage_keep * shape[0]))+1)/2.)  # Compute the size of the window to keep
half_x_keep = int(((int(percentage_keep * shape[1]))+1)/2.)
fft_thresh[middle_y - half_y_keep: middle_y + half_y_keep, middle_x - half_x_keep: middle_x + half_x_keep] = \
    fft_noisy[middle_y - half_y_keep: middle_y + half_y_keep, middle_x - half_x_keep: middle_x + half_x_keep]

# Inverse transform
im_denoised = ifft2(ifftshift(fft_thresh)).real
im_denoised = np.clip(im_denoised, 0., 1.)  # Ensure the values stay in [0., 1.]

print_psnr_ssim(im_denoised, im1_gray, 'Denoised')

# Display
if True:
    plt.subplot(3, 2, 1)
    plt.imshow(im1_gray, cmap='gray')
    plt.title('Clean Image')

    plt.subplot(3, 2, 2)
    plt.imshow(np.abs(fft_clean), norm=LogNorm())
    plt.title('Clean Spectrum')
    plt.colorbar()

    plt.subplot(3, 2, 3)
    plt.imshow(im_noise_lib, cmap='gray')
    plt.title('Noisy Image')

    plt.subplot(3, 2, 4)
    plt.imshow(np.abs(fft_noisy), norm=LogNorm())
    plt.title('Noisy Spectrum')
    plt.colorbar()

    plt.subplot(3, 2, 5)
    plt.imshow(im_denoised, cmap='gray')
    plt.title('Denoised Image')

    plt.subplot(3, 2, 6)
    plt.imshow(np.abs(fft_thresh), norm=LogNorm(vmin=0.1))
    plt.title('Denoised Spectrum')
    plt.colorbar()

    plt.show()


