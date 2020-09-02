"""
Authors: Florian Lemarchand, Maxime Pelcat
Date: 2020
"""

# ========= Imports =========
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.util import random_noise


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

# Noise im1 with Gaussian Noise
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


