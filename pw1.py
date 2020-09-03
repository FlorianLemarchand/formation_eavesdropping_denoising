"""
Authors: Florian Lemarchand, Maxime Pelcat
Date: 2020
"""

# ========= Imports =========
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.util import random_noise, img_as_ubyte, img_as_float


# Read image
im1 = imread('data/im1.jpg')
im2 = imread('data/im2.jpg')

# Print the shapes of the images
print('Shape of im1: {}'.format(im1.shape))
print('Shape of im2: {}'.format(im2.shape))

# Display the images
plt.subplot(2,1,1)
plt.title('Image 1')
plt.imshow(im1)
plt.subplot(2,1,2)
plt.title('Image 2')
plt.imshow(im2)
plt.show()
plt.close()

# Crop patchs
crop_1 = im1[0:150, 0:150]
crop_2 = im2[0:100, 150:250]

plt.subplot(2,2,1)
plt.title('Image 1')
plt.imshow(im1)
plt.subplot(2,2,2)
plt.title('Crop 1')
plt.imshow(crop_1)
plt.subplot(2,2,3)
plt.title('Image 2')
plt.imshow(im2)
plt.subplot(2,2,4)
plt.title('Crop 2')
plt.imshow(crop_2)
plt.show()
plt.close()

# Separate Image Channels
red_channel_1 = im1[:, :, 0]
green_channel_2 = im2[:, :, 1]

plt.subplot(2,2,1)
plt.title('Image 1')
plt.imshow(im1)
plt.subplot(2,2,2)
plt.title('Red Channel')
plt.imshow(red_channel_1, cmap='gray')
plt.subplot(2,2,3)
plt.title('Image 2')
plt.imshow(im2)
plt.subplot(2,2,4)
plt.title('Green Channel')
plt.imshow(green_channel_2, cmap='gray')
plt.show()
plt.close()

# Noise im1 with Gaussian Noise --> N(0,50)
# https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise
mean = 0
sigma = 50
sigma = sigma / 255.  # Normalize sigma the standard deviation for 8-bit pixel values
var = np.square(sigma)  # Get variance from sigma
im1_noisy = random_noise(im1, 'gaussian', mean=0, var=var, seed=0, clip=True)

plt.subplot(1,2,1)
plt.title('Image 1')
plt.imshow(im1)
plt.subplot(1,2,2)
plt.title('Noisy Image 1')
plt.imshow(im1_noisy)
plt.show()
plt.close()

# Noise im1 with different sigmas
sigmas = range(20, 161, 20)

ncol = 3
nrow = int((len(sigmas) + 2) / ncol)

print('{} image to diplay on {} columns and {} rows'.format(len(sigmas) + 1, ncol, nrow))

plt.subplot(nrow, ncol, 1)
plt.title('Image 1')
plt.imshow(im1)
for i, sigma in enumerate(sigmas):
    var = np.square(sigma / 255.)  # Compute variance from sigma
    im_noise = random_noise(im1, 'gaussian', mean=0, var=var, seed=0, clip=True)  # Noise the image

    plt.subplot(nrow, ncol, i + 2)
    plt.title('Sigma = {}'.format(sigma))
    plt.imshow(im_noise)

plt.show()
plt.close()

# Display histograms of clean and noisy images
im1_heigth, im1_width, im1_channels = im1.shape

# im1_heigth = im1.shape[0]
# im1_width = im1.shape[1]
# im1_channels = im1.shape[2]

vec_im1 = np.reshape(im1, (im1_heigth * im1_width, im1_channels))  # Vectorize im1
vec_im1_noisy = np.reshape(img_as_ubyte(im1_noisy), (im1_heigth * im1_width, im1_channels))  # Vectorize im1_noisy


plt.subplot(3,3,1)
plt.title('Image 1')
plt.imshow(im1)
plt.subplot(3,3,3)
plt.title('Noisy Image 1')
plt.imshow(im1_noisy)

plt.subplot(3,3,4)
plt.title('Hist R')
plt.hist(vec_im1[:, 0], bins=256)
plt.subplot(3,3,5)
plt.title('Hist G')
plt.hist(vec_im1[:, 1], bins=256)
plt.subplot(3,3,6)
plt.title('Hist B')
plt.hist(vec_im1[:, 2], bins=256)

plt.subplot(3,3,7)
plt.title('Hist R Noisy')
plt.hist(vec_im1_noisy[:, 0], bins=256, range=(1, 254))
plt.subplot(3,3,8)
plt.title('Hist G Noisy')
plt.hist(vec_im1_noisy[:, 1], bins=256, range=(1, 254))
plt.subplot(3,3,9)
plt.title('Hist B Noisy')
plt.hist(vec_im1_noisy[:, 2], bins=256, range=(1, 254))

plt.show()
plt.close()
