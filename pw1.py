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
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter, median_filter, convolve


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
sigma_float = sigma/255.
variance = np.square(sigma_float) # variance = np.square(std)

# Add AWGN sigma 50 to image
im_noise_lib = random_noise(im1_gray, 'gaussian', mean=0., seed=0, var=variance, clip=True)



# # Transform the image
# dct_clean = dct(im1_gray)
# vec_dct_clean = np.reshape(dct_clean, dct_clean.size)  # Vectorize noise_map
#
# dct_noisy = dct(im_noise_lib)
# vec_dct_noisy = np.reshape(dct_noisy, dct_noisy.size)  # Vectorize noise_map
#
# # Set low right part of the image to 0
# dct_thresh = dct_noisy
# vec_dct_thresh = np.reshape(dct_thresh, dct_thresh.size)  # Vectorize noise_map
#
# height, width = dct_noisy.shape
# half_height, half_width = int(height/2), int(width/2)
# std = np.std(dct_noisy)
#
# print(np.min(dct_clean), np.max(dct_clean), np.mean(dct_clean))
#
# print(np.min(dct_noisy), np.max(dct_noisy), np.mean(dct_noisy))
#
# for i in range(0, height):
#     for j in range(0, width):
#         if i > half_height or j > half_width:
#             dct_thresh[i, j] = np.clip(dct_thresh[i, j], -3 * std, 3 * std)
# # print(std)
# # print(np.min(dct_noisy), np.max(dct_noisy),np.mean(dct_noisy))
# # dct_thresh = np.clip(dct_noisy, -3 * std, 3 * std)
# # print(np.min(dct_thresh), np.max(dct_thresh), np.mean(dct_thresh))
#
# # Inverse transform
# im_denoised = idct(dct_thresh)
# im_denoised = np.clip(im_denoised, 0. , 1.)
#
# # Display
# if True:
#     plt.subplot(3, 2, 1)
#     plt.imshow(im1_gray, cmap='gray')
#     plt.subplot(3, 2, 2)
#     plt.imshow(dct_clean, cmap='gray')
#     # plt.hist(vec_dct_clean, bins=1000)
#
#     plt.subplot(3, 2, 3)
#     plt.imshow(im_noise_lib, cmap='gray')
#     plt.subplot(3, 2, 4)
#     plt.imshow(dct_noisy, cmap='gray')
#     # plt.hist(vec_dct_noisy, bins=1000)
#
#     plt.subplot(3, 2, 5)
#     plt.imshow(im_denoised, cmap='gray')
#     plt.subplot(3, 2, 6)
#     plt.imshow(dct_thresh, cmap='gray')
#     # plt.hist(vec_dct_thresh, bins=1000)
#
#     plt.show()

# im_denoised = median_filter(im_noise_lib, 3)

# Measure the quality of the noisy images with respect to the clean image

print_psnr_ssim(im_noise_lib, im1_gray, 'Noisy')
# Filter
def mean_filter(input, kernel_size):
  kernel = np.ones((kernel_size, kernel_size))
  output = 1. / np.square(kernel_size) * convolve(input, kernel, mode='constant', cval=0.)
  return output


def approx_gauss_filter_5(input):
    kernel = [[1, 4, 6, 4, 1],
              [4, 16, 24, 16, 4],
              [1, 24, 36, 24, 6],
              [4, 16, 24, 16, 4],
              [1, 4, 6, 4, 1]]
    sum = np.sum(kernel)
    output = 1 / sum * convolve(input, kernel)
    return output


def filter_loop(noisy_image, kernel, padding_type='zeros'):
    height, width = noisy_image.shape
    kernel_size = kernel.shape[0]

    # Initialize the output array
    output = np.zeros(noisy_image.shape)

    # Generate padding
    padding_size = int(kernel_size/2)

    if padding_type is 'zeros':
        padded_input = np.zeros((height + 2 * padding_size, width + 2 * padding_size))
    elif padding_type is 'ones':
        padded_input = np.ones((height + 2 * padding_size, width + 2 * padding_size))

    padded_input[padding_size:padding_size+height, padding_size:padding_size+width] = noisy_image

    # Loop over the image
    for i in range(0, height):
        for j in range(0, width):
            output[i, j] = np.sum(np.multiply(kernel, padded_input[i:i+kernel_size, j:j+kernel_size]))

    return output


mean_denoised = mean_filter(im_noise_lib, 3)
gauss_denoised = approx_gauss_filter_5(im_noise_lib)
hand_denoised = filter_loop(im_noise_lib, np.ones((3, 3)) / 9.)

# Measure the quality of the denoised images with respect to the clean image
mean_denoised_psnr = np.round(peak_signal_noise_ratio(mean_denoised, im1_gray), 2)
mean_denoised_ssim = np.round(structural_similarity(mean_denoised, im1_gray), 2)
print('Mean Denoised: PSNR: {} / SSIM: {}'.format(mean_denoised_psnr, mean_denoised_ssim))

hand_mean_denoised_psnr = np.round(peak_signal_noise_ratio(hand_denoised, im1_gray), 2)
hand_mean_denoised_ssim = np.round(structural_similarity(hand_denoised, im1_gray), 2)
print('Hand Mean Denoised: PSNR: {} / SSIM: {}'.format(hand_mean_denoised_psnr, hand_mean_denoised_ssim))

gauss_denoised_psnr = np.round(peak_signal_noise_ratio(gauss_denoised, im1_gray), 2)
gauss_denoised_ssim = np.round(structural_similarity(gauss_denoised, im1_gray), 2)
print('Gauss Denoised: PSNR: {} / SSIM: {}'.format(gauss_denoised_psnr, gauss_denoised_ssim))
