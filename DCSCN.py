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
        self.checkpoint_dir = flags.checkpoint_dir
                                                # Directory for checkpoints
        self.tf_log_dir = flags.tf_log_dir      # Directory for tensorboard log

        # Debuging or Logging
        self.debug = flags.debug                # Display each calculated MSE and weight variables
        self.save_loss = flags.save_loss        # Save loss
        self.save_weights = flags.save_weights  # Save weights and biases
        self.save_images = flags.save_images    # Save CNN weights as images
        self.save_images_num = flags.save_images_num
                                                # Number of CNN images saved
        self.log_weight_image_num = 32

        # initialize variables
        self.name = self.get_model_name(model_name)
        self.batch_input = self.batch_num * [None]
        self.batch_input_quad = np.zeros(
            shape=[self.batch_num, self.batch_image_size, self.batch_image_size, self.scale * self.scale]
        )

        self.batch_true_quad = np.zeros(
            shape=[self.batch_num, self.batch_image_size, self.batch_image_size, self.scale * self.scale]
        )
        self.receptive_fields = 2 * self.layers + self.cnn_size - 2
        self.complexity = 0

        # initialize environment
        util.make_dir(self.checkpoint_dir)
        util.make_dir(flags.graph_dir)
        util.make_dir(self.tf_log_dir)
        if flags.initialise_tf_log:
            util.clean_dir(self.tf_log_dir)
        util.set_logging(flags.log_filename, stream_log_level=logging.INFO, file_log_level=logging.INFO,
                         tf_log_level=tf.logging.WARN)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        self.init_train_step()

        logging.info("\nDCSCN -------------------------------")
        logging.info("{} [{}]".format(util.get_now_date(), self.name))

    def get_model_name(self, model_name):
        if model_name is "":
            name = "dcscn_L{:d}_F{:d}".format(self.layers, self.filters)
            if self.min_filters != 0:
                name += "to{:D}".format(self.min_filters)
            if self.filters_decay_gamma != 1.0:
                name += "_G{:2.2f}".format(self.filters_decay_gamma)
            if self.cnn_size != 3:
                name += "_C{:d}".format(self.cnn_size)
            if self.scale != 2:
                name += "_S{:d}".format(self.scale)
            if self.nin:
                name += "_NIN"
            if self.nin_filters != 0:
                name += "_A{:d}".format(self.nin_filters)
                if self.nin_filters2 != self.nin_filters2 // 2:
                    name += "_B{:d}".format(self.nin_filters2)

            if self.bicubic_init:
                name += "_BI"
            if self.dropout != 1.0:
                name += "_D{:0.2f}".format(self.dropout)
            if self.max_value != 255.0:
                name += "_M{:2.1f}".format(self.max_value)
            if self.activator != "relu":
                name += "_{}".format(self.activator)
            if self.dataset != "yang91":
                name += "_" + self.dataset
            if self.batch_image_size != 32:
                name += "_B{:d}".format(self.batch_image_size)
            if self.last_cnn_size != 1:
                name += "_L{:d}".format(self.last_cnn_size)
        else:
            name = "dcscn_{}".format(model_name)

        return name

    def load_datasets(self, target, data_dir, batch_dir, batch_image_size, stride_size=0):

        print("Loading datasets for [%s]..." % target)
        util.make_dir(batch_dir)

        if stride_size == 0:
            stride_size = batch_image_size // 2

        if self.bicubic_init:
            resampling_method = "bicubic"
        else:
            resampling_method = "nearest"

        datasets = loader.DataSets(self.scale, batch_image_size, stride_size, channels=self.channels,
                                   jpeg_mode=self.jpeg_mode, max_value=self.max_value,
                                   resampling_method=resampling_method)

        if not datasets.is_batch_exist(batch_dir):
            datasets.build_batch(data_dir, batch_dir)

        if target == "training":
            datasets.load_batch_train(batch_dir)
            self.train = datasets
        else:
            datasets.load_batch_test(batch_dir)
            self.test = datasets

    def init_epoch_index(self):

        self.batch_index = random.sample(range(0, self.train.input.count), self.train.input.count)
        self.index_in_epoch = 0
        self.training_psnr_sum = 0
        self.training_step = 0

    def build_input_batch(self):

        for i in range(self.batch_num):
            self.batch_input[i], self.batch_input_bicubic[i], self.batch_true[i] = self.train.load_batch_image(
                self.max_value)

    def conv2d(self, x, w, stride, bias=None, activator=None, leaky_relu_alpha=0.1, name=""):
        conv = tf.nn.conv2d(x, w, strides=[stride, stride, 1, 1], padding="SAME", name=name + "_conv")

        self.complexity += int(w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3])

        if bias is not None:
            conv = tf.add(conv, bias, name=name + "_add")

            self.complexity += int(bias.shape[0])

        if activator is not None:
            if activator == "relu":
                conv = tf.nn.relu(conv, name=name + "_relu")
            elif activator == "sigmoid":
                conv = tf.nn.sigmoid(conv, name=name + "_sigmoid")
            elif activator == "tanh":
                conv = tf.nn.tanh(conv, name=name + "_tanh")
            elif activator == "leaky_relu":
                conv = tf.maximum(conv, leaky_relu_alpha * conv, name=name + "_leaky")
            elif activator == "prelu":
                with tf.variable_scope("prelu"):
                    alphas = tf.Variable(tf.constant(0.1, shape=[w.get_shape()[3]]), name=name + "_prelu")
                    if self.save_weights:
                        util.add_summaries("prelu_alpha", self.name, alphas, save_stddev=False, save_mean=False)
                    conv = tf.nn.relu(conv) + tf.multiply(alphas, (conv - tf.abs(conv))) * 0.5
            else:
                raise NameError('Not implemented activator:%s' % activator)
            self.complexity += int(bias.shape[0])
        return conv

    def build_conv_and_bias(self, name, input_tensor, cnn_size, input_feature_num, output_feature_num,
                            use_activation=True, use_dropout=True):
        with tf.variable_scope(name):
            w = util.weight([cnn_size, cnn_size, input_feature_num, output_feature_num],
                            stddev=self.weight_dev, name="conv_W", initializer=self.initializer)

            b = util.bias([output_feature_num], name="conv_B")

            h = self.conv2d(input_tensor, w, self.cnn_stride, bias=b,
                            activator=self.activator if use_activation else None,
                            name=name)

            if use_dropout and self.dropout != 1.e0:
                h = tf.nn.dropout(h, self.dropout_input, name="dropout")

            if self.save_weights:
                util.add_summaries("weight", self.name, w, save_stddev=True, save_mean=True)
                util.add_summaries("bias", self.name, b, save_stddev=True, save_mean=True)

            if self.save_images and cnn_size > 1 and input_feature_num == 1:
                weight_transposed = tf.transpose(w, [3, 0, 1, 2])
                with tf.name_scope("image"):
                    tf.summary.image(self.name, weight_transposed, max_outputs=self.log_weight_image_num)

        return w, b, h

    def build_conv(self, name, input_tensor, cnn_size, input_feature_num, output_feature_num):
        with tf.variable_scope(name):
            w = util.weight([cnn_size, cnn_size, input_feature_num, output_feature_num],
                            stddev=self.weight_dev, name="conv_W", initializer=self.initializer)

            h = self.conv2d(input_tensor, w, self.cnn_stride, bias=None, activator=None, name=name)

            if self.save_weights:
                util.add_summaries("weight", self.name, w, save_stddev=True, save_mean=True)

            if self.save_images and cnn_size > 1 and input_feature_num == 1:
                weight_transposed = tf.transpose(w, [3, 0, 1, 2])
                with tf.name_scope("image"):
                    tf.summary.image(self.name, weight_transposed, max_outputs=self.log_weight_image_num)

        return w, h

    def build_input_batch(self, batch_dir):

        for i in range(self.batch_num):
            if self.index_in_epoch >= self.train.input.count:
                self.init_epoch_index()
                self.epochs_completed += 1

            image_no = self.batch_index[self.index_in_epoch]
            self.batch_input[i] = util.load_image(batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.bmp" % image_no,
                                                  print_console=False)
            batch_input_quad = util.load_image(batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.bmp" % image_no,
                                               print_console=False)
            loader.convert_to_multi_channel_image(self.batch_input_quad[i], batch_input_quad, self.scale)
            batch_true_quad = util.load_image(batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.bmp" % image_no,
                                              print_console=False)
            loader.convert_to_multi_channel_image(self.batch_true_quad[i], batch_true_quad, self.scale)
            self.index_in_epoch += 1

    def init_train_step(self):
        self.lr = self.initial_lr
        self.csv_epochs = []
        self.csv_psnr = []
        self.csv_training_psnr = []
        self.epochs_completed = 0
        self.min_validation_mse = -1
        self.min_validation_epoch = -1
        self.step = 0

        self.start_time = time.time()

    def end_train_step(self):
        self.total_time = time.time() - self.start_time

    def print_steps_completed(self, output_to_logging=False):
        if self.step == 0:
            return
        processing_time = self.total_time / self.step
        h = self.total_time // (60 * 60)
        m = (self.total_time - h * 60 * 60) // 60
        s = (self.total_time - h * 60 * 60 - m * 60)

        status = "Finished at Total Epoch:%d Steps:%s Time:%02d:%02d:%02d (%2.3fsec/step)" % (
            self.epochs_completed, "{:,}".format(self.step), h, m, s, processing_time)

        if output_to_logging:
            logging.info(status)

        else:
            print(status)
