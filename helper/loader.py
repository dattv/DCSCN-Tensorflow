"""
    Function for loading converting data
"""

import configparser
import logging
import os
import random

import numpy as np
from scipy import misc

from helper import utility as util

INPUT_IMAGE_IDR = "input"
INTERPOLATED_IMAGE_DIR = "interpolated"
TRUE_IMAGE_DIR = "true"

def build_image_set(file_path, chanels=1, scale=1, convert_ycbcr=True, resampleing_method="bicubic",
                    print_console=True):
    true_image = util.set_image_alignment(util.load_image(file_path, print_console=print_console), scale)

    if chanels == 1 and true_image.shape[2] == 3 and convert_ycbcr:
        true_image = util.convert_rgb_to_y(true_image)

