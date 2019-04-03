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

