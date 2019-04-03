import logging
import os
import shutil

import tensorflow as tf

from helper import utility as util


class TensorflowGraph:
    def __init__(self, flags):
        # graph setting
        self.dropout_rate = flags.dropout_rate
        self.activator = flags.activator
        self.batch_norm = flags.batch_norm
        self.cnn_size = flags.cnn_size
        self.cnn_stride = 1
        self.initializer = flags.initializer
        self.weight_dev = flags.seight_dev

        # graph place holder
        self.is_training = None
        self.dropout = False
        self.saver = None
        self.summary_op = None
        self.train_writer = None
        self.test_writer = None

        # Debugging or Logging
