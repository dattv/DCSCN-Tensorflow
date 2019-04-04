"""
    Function for sharing arguments and their default values
    Reference: https://github.com/jiny2001/dcscn-super-resolution/blob/master/helper/args.py
"""

import sys
import numpy as np
import tensorflow as tf
from numpy.f2py.auxfuncs import F2PYError

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model
flags.DEFINE_integer("filters", 96, "Number of CNN Filters")
flags.DEFINE_integer("min_filters", 32, "Number of the last CNN Filters")
flags.DEFINE_integer("nin_filters", 64, "Number of Filters of A1 in Reconstruction network")
flags.DEFINE_integer("nin_filters2", 0, "Number of Filters of B1 and B2 in Reconstruction. If 0, it will be half of A1")
flags.DEFINE_integer("cnn_size", 3, "Size of CNN filters")
flags.DEFINE_integer("last_cnn_size", 1, "Size of Last CNN filters")
flags.DEFINE_integer("layers", 7, "Number of layers of CNNs")
flags.DEFINE_boolean("nin", True, "Use Network In Network")
flags.DEFINE_boolean("bicubic_init", True, "make bicubic interpolation values as initial input of x2")
flags.DEFINE_float("dropout", 0.8, "For dropout value for  value. Don't use if it's 1.0.")
flags.DEFINE_string("activator", "prelu", "Activator. can be [relu, leaky_relu, prelu, sigmoid, tanh]")
flags.DEFINE_float("filters_decay_gamma", 1.2, "Gamma")

# Training
flags.DEFINE_string("initializer", "he", "Initializer for weights can be [uniform, stddev, xavier, he, identity, zero]")
flags.DEFINE_float("weight_dev", 0.01, "Initial weight stddev (won't be used when you use he or xavier initializer)")
flags.DEFINE_float("l2_decay", 0.0001, "l2_decay")
flags.DEFINE_string("optimizer", "adam", "Optimizer can be [gd, momentum, adadelta, adagrad, adam, rmsprop]")
flags.DEFINE_float("beta1", 0.1, "Beta1 for adam optimizer")
flags.DEFINE_float("beta2", 0.1, "Beta2 for adam optimizer")
flags.DEFINE_float("momentum", 0.9, "Momentum for momentum optimizer and rmsprop optimizer")
flags.DEFINE_integer("batch_num", 40, "Number of mini-batch images for training")
flags.DEFINE_integer("batch_image_size", 32, "Image size for mini-batch")
flags.DEFINE_integer("stride_size", 0, "Stride size for mini-batch. If it is 0, use half of batch_image_size")

# Learning Rate Control for Training
flags.DEFINE_float("initial_lr", 0.001, "Initial learning rate")
flags.DEFINE_float("lr_decay", 0.5, "Learning rate decay rate when it does not reduced during specific epoch")
flags.DEFINE_integer("lr_decay_epoch", 5, "Decay learning rate when loss does not decrease (5)")
flags.DEFINE_float("end_lr", 2e-5, "Training end learning rate (2e-5")

# Dataset or Others
flags.DEFINE_string("test_dataset", "set5", "Directory of Test dataset [set5, set14, bsd100, urban100]")
flags.DEFINE_string("dataset", "yang91", "Training dataset dir. [yang91, general100, bsd200]")
flags.DEFINE_integer("tests", 1, "Number of training tests")

# Image Processing
flags.DEFINE_integer("scale", 2, "Scale factor for Super Resolution (can be 2 or more)")
flags.DEFINE_float("max_value", 255, "For normalize image pixel value")
flags.DEFINE_integer("channels", 1, "Number of image channels used. Use only Y of YCbCr when channels=1.")
flags.DEFINE_boolean("jpeg_mode", False, "Turn on or off jpeg mode when converting from rgb to ycbcr")

# Environment (all directory name should not contain '/' after )
flags.DEFINE_string("checkpoint_dir", "models", "Directory for checkpoints")
flags.DEFINE_string("graph_dir", "graphs", "Directory for graphs")
flags.DEFINE_string("data_dir", "data", "Directory for original images")
flags.DEFINE_string("batch_dir", "batch_data", "Directory for training batch images")
flags.DEFINE_string("output_dir", "output", "Directory for output test images")
flags.DEFINE_string("tf_log_dir", "tf_log", "Directory for tensorboard log")
flags.DEFINE_string("log_filename", "log.txt", "log filename")
flags.DEFINE_string("model_name", "", "model name for save files and tensorboard log")
flags.DEFINE_string("load_model_name", "", "Filename of model loading before start [filename or 'default']")

# Debugging or Logging
flags.DEFINE_boolean("debug", False, "Display each calculated MSE and weight variables")
flags.DEFINE_boolean("initialise_tf_log", True, "Clear all tensorboard log before start")
flags.DEFINE_boolean("save_loss", True, "Save loss")
flags.DEFINE_boolean("save_weights", False, "Save weights and biases")
flags.DEFINE_boolean("save_images", False, "Save CNN weights as images")
flags.DEFINE_integer("save_images_num", 10, "Number of CNN images saved")

def get():
    print("Python Interpreter version: {}".format(sys.version[:3]))
    print("Tensorflow version: {}".format(tf.__version__))
    print("Numpy version: {}".format(np.__version__))

    # check which library you are using
    return FLAGS
