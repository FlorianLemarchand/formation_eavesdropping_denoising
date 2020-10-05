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
from skimage.measure import compare_psnr, compare_mse, compare_ssim

from skimage.util import random_noise, img_as_ubyte, img_as_float


# ========= Write your code hereunder =========
