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
        self.enable_log = flags.enable_log
        self.save_weights = flags.save_weights and flags.enable_log

        self.save_images = flags.save_images and flags.enable_log

        self.save_images_num = flags.save_images_num
        self.save_meta_data = flags.save_meta_data and flags.enable_log

        self.log_seights_image_num = 32

        # Environment (all directory name should not contain '/' after)
        self.checkpoint_dir = flags.checkpoint_dir
        self.tf_log_dir = flags.tf_log_dir

        # Status / attributes
        self.Weights = []
        self.Bias = []
        self.features = ""
        self.H = []
        self.receptive_fields = 0
        self.complexity = 0
        self.pix_per_input = 1

        self.init_session(flags.gpu_device_id)

    def init_session(self, device_id):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True ## just for use the necesary memory of GPU
        config.gpu_options.visible_device_list = str(device_id) ## this values depends of numbers of GPUs

        print("Session and graph initialized. ")
        self.sess = tf.InterativeSession(config=config, graph=tf.Graph())

    def init_all_variables(self):
        self.sess.run(tf.global_variables_initializer())
        print("Model initialized.")

    def build_activtor(self, input_tensor, features, activator="", leaky_relu_alpha=0.1, base_name=""):

        features = int(features)
        if activator is None or "":
            return
        elif activator == "relu":
            output = tf.nn.relu(input_tensor, name=base_name + "_relu")
        elif activator == "sigmoid":
            output = tf.nn.sigmoid(input_tensor, name=base_name + "_relu")
        elif activator == "tanh":
            output = tf.nn.tanh(input_tensor, name=base_name + "_tanh")
        elif activator == "leaky_relu":
            output = tf.maximum(input_tensor, leaky_relu_alpha * input_tensor, name=base_name + "_leaky")
        elif activator == "prelu":
            with tf.variable_scope("prelu"):
                alphas = tf.Variable(tf.constant(0.1, shape=[features]), name=base_name + "_prelu")
                if self.save_weights:
                    util.add_summaries("prelu_alpha", self.name, alphas, save_stddev=False, save_mean=False)
                output = tf.nn.relu(input_tensor) + tf.multiply(alphas, (input_tensor - tf.abs(input_tensor))) * 0.5
        elif activator == "selu":
            output = tf.nn.selu(input_tensor, base_name=base_name + "_selu")
        else:
            raise NameError("Not implemented activator: {}".format(activator))

        self.complexity += (self.pix_per_input * features)
        return output

    def conv2d(self, input_tensor, w, stride, bias=None, use_batch_norm=False, name=""):
        output = tf.nn.conv2d(input_tensor, w, strides=[1, stride, stride, 1], padding="SAME", name=name + "_conv")
        self.complexity += self.pix_per_input + int(w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3])

        if bias is not None:
            output = tf.add(output, bias, name= name + "_add")
            self.complexity += self.pix_per_input * int(bias.shape[0])

        if use_batch_norm:
            output = tf.layers.batch_normalization(output, training=self.is_training, name="BN")

        return output

