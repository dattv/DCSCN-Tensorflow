"""

"""
import logging
import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from helper import loader, utility as util

matplotlib.use("agg")

INPUT_IMAGE_DIR = "input"
INTERPOLATED_IMAGE_DIR = "interpolated"
TRUE_IMAGE_DIR = "true"

class SuperResolution:
    def __init__(self, flags, model_name=""):

        # model parameters
        self.filters = flags.filters            # number of filters in first feature extraction CNNs
        self.min_filters = flags.min_filters    # number of filters in last feature extraction CNNs
        self.nin_filters = flags.nin_filters    # Number of CNNs filters in A1 at reconstruction network
        self.nin_filters2 = flags.nin_filters2 if flags.nin_filters2 != 0 else flags.nin_filters //2
                                                # Number of CNNs filters in B1 and B2 at reconsruction network
        self.cnn_size = flags.cnn_size          # size of CNNs features
        self.last_cnn_size = flags.last_cnn_size# Size of Last CNN filters
        self.cnn_stride = 1
        self.layers = flags.layers              # Number of layers of CNNs
        self.nin = flags.nin                    # Use Network In Network
        self.bicubic_init = flags.bicubic_init  # make bicubic interpolation values as initial input of x2
        self.dropout = flags.dropout            # For dropout value for  value. Don't use if it's 1.0.
        self.activator = flags.activator        # Activator. can be [relu, leaky_relu, prelu, sigmoid, tanh]
        self.filters_decay_gamma = flags.filters_decay_gamma
                                                # Gamma
        # Training parameters
        self.initializer = flags.initializer    # Initializer for weights can be [uniform, stddev, xavier, he, identity, zero]
        self.weight_dev = flags.weight_dev      # Initial weight stddev (won't be used when you use he or xavier initializer)
        self.l2_decay = flags.l2_decay          # l2_decay
        self.optimizer = flags.optimizer        # Optimizer can be [gd, momentum, adadelta, adagrad, adam, rmsprop]
        self.beta1 = flags.beta1                # Beta1 for adam optimizer
        self.beta2 = flags.beta2                # Beta2 of adam optimizer
        self.momentum = flags.momentum          # Momentum for momentum optimizer and rmsprop optimizer
        self.batch_num = flags.batch_num        # Number of mini-batch images for training
        self.batch_image_size = flags.image_size# mage size for mini-batch
        if flags.stride_size == 0:
            self.stride_size = flags.batch_image_size // 2
        else:
            self.stride_size = flags.stride_size

        # Learning rate control for training
        self.initial_lr = flags.initial_lr      # Initial learning rate
        self.lr_decay = flags.lr_decay          # Learning rate decay rate when it does not reduced during specific epoch
        self.lr_decay_epoch = flags.lr_decay_epoch
                                                # Decay learning rate when loss does not decrease (5)
        # Dataset or Others
        self.dataset = flags.dataset            # Training dataset dir. [yang91, general100, bsd200]
        self.test_dataset = flags.test_dataset  # Directory of Test dataset [set5, set14, bsd100, urban100]

        # Image Processing Parameters
        self.scale = flags.scale                # Scale factor for Super Resolution (can be 2 or more)
        self.max_value = flags.max_value        # For normalize image pixel value
        self.chanels = flags.chanels            # Number of image channels used. Use only Y of YCbCr when channels=1.
        self.jpeg_mode = flags.jpeg_model       # Turn on or off jpeg mode when converting from rgb to ycbcr
        self.output_channels = self.scale * self.scale
                                                #
        # Environment



