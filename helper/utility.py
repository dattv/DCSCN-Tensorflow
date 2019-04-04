"""
    Utility functions
"""

import logging
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import misc
import math
import datetime
from skimage.measure import compare_psnr, compare_ssim


class Timer:
    def __init__(self, timer_count=100):
        self.times = np.zeros(timer_count)
        self.start_times = np.zeros(timer_count)
        self.counts = np.zeros(timer_count)
        self.timer_count = timer_count

    def start(self, timer_id):
        self.start_times[timer_id] = time.time()

    def end(self, timer_id):
        self.times[timer_id] += time.time() - self.start_times[timer_id]
        self.counts[timer_id] += 1

    def print(self):
        for i in range(self.timer_count):
            if self.counts[i] > 0:
                total = 0
                print("Average of %d: %s[ms]" % (i, "{:,}".format(self.times[i] * 1000 / self.counts[i])))
                total += self.times[i]
                print("Total of %d: %s" % (i, "{:,}".format(total)))


# utilities for save / load

class LoadError(Exception):
    def __init__(self, message):
        self.message = message


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_files_in_directory(path):
    if not path.endswith('/'):
        path = path + "/"
    file_list = [path + f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]
    return file_list


def remove_generic(path, __func__):
    try:
        __func__(path)
    except OSError as error:
        print("OS error: {0}".format(error))


def clean_dir(path):
    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    for x in files:
        full_path = os.path.join(path, x)
        if os.path.isfile(full_path):
            f = os.remove
            remove_generic(full_path, f)
        elif os.path.isdir(full_path):
            clean_dir(full_path)
            f = os.rmdir
            remove_generic(full_path, f)


def set_logging(filename, stream_log_level, file_log_level, tf_log_level):
    stream_log = logging.StreamHandler()
    stream_log.setLevel(stream_log_level)

    file_log = logging.FileHandler(filename=filename)
    file_log.setLevel(file_log_level)

    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(stream_log)
    logger.addHandler(file_log)
    logger.setLevel(min(stream_log_level, file_log_level))

    tf.logging.set_verbosity(tf_log_level)

    # optimizing logging
    logging._srcfile = None
    logging.logThreads = 0
    logging.logProcesses = 0


def save_image(filename, image, print_console=True):
    if len(image.shape) >= 3 and image.shape[2] == 1:
        image = image.reshape(image.shape[0], image.shape[1])

    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    image = misc.toimage(image, cmin=0, cmax=255)  # to avoid range rescaling
    misc.imsave(filename, image)

    if print_console:
        print("Saved [%s]" % filename)


def save_image_data(filename, image):
    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    np.save(filename, image)
    print("Saved [%s]" % filename)


def convert_rgb_to_y(image, jpeg_mode=True, max_value=255.0):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114]])
        y_image = image.dot(xform.T)
    else:
        xform = np.array([[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0]])
        y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

    return y_image


def convert_rgb_to_ycbcr(image, jpeg_mode=True, max_value=255):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, [1, 2]] += max_value / 2
    else:
        xform = np.array(
            [[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
             [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
        ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)

    return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image, jpeg_mode=True, max_value=255.0):
    if len(y_image.shape) <= 2:
        y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image[:, :, 0]
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=jpeg_mode, max_value=max_value)


def convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=True, max_value=255.0):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    if jpeg_mode:
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array([[1, 0, 1.402], [1, - 0.344, - 0.714], [1, 1.772, 0]])
        rgb_image = rgb_image.dot(xform.T)
    else:
        rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - (16.0 * max_value / 256.0)
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array(
            [[max_value / 219.0, 0, max_value * 0.701 / 112.0],
             [max_value / 219, - max_value * 0.886 * 0.114 / (112 * 0.587),
              - max_value * 0.701 * 0.299 / (112 * 0.587)],
             [max_value / 219.0, max_value * 0.886 / 112.0, 0]])
        rgb_image = rgb_image.dot(xform.T)

    return rgb_image


def set_image_alignment(image, alignment):
    alignment = int(alignment)
    width, height = image.shape[1], image.shape[0]
    width = (width // alignment) * alignment
    height = (height // alignment) * alignment

    if image.shape[1] != width or image.shape[0] != height:
        image = image[:height, :width, :]

    if len(image.shape) >= 3 and image.shape[2] >= 4:
        image = image[:, :, 0:3]

    return image


def resize_image_by_bicubic(image, scale):
    size = [int(image.shape[0] * scale), int(image.shape[1] * scale)]
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    tf_image = tf.image.resize_bicubic(image, size=size)
    image = tf_image.eval()
    return image.reshape(image.shape[1], image.shape[2], image.shape[3])


def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest":
        method = Image.NEAREST
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # the image may has an alpha channel
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image


def load_image(filename, width=0, height=0, channels=0, alignment=0, print_console=True):
    if not os.path.isfile(filename):
        raise LoadError("File not found [%s]" % filename)
    image = misc.imread(filename)

    if len(image.shape) == 2:
        image = image.reshape(image.shape[0], image.shape[1], 1)
    if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
        raise LoadError("Attributes mismatch")
    if channels != 0 and image.shape[2] != channels:
        raise LoadError("Attributes mismatch")
    if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
        raise LoadError("Attributes mismatch")

    if print_console:
        print("Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2]))
    return image


def load_image_data(filename, width=0, height=0, channels=0, alignment=0, print_console=True):
    if not os.path.isfile(filename):
        raise LoadError("File not found")
    image = np.load(filename)

    if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
        raise LoadError("Attributes mismatch")
    if channels != 0 and image.shape[2] != channels:
        raise LoadError("Attributes mismatch")
    if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
        raise LoadError("Attributes mismatch")

    if print_console:
        print("Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2]))
    return image


def get_split_images(image, window_size, stride=None, enable_duplicate=True):
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.reshape(image.shape[0], image.shape[1])

    window_size = int(window_size)
    size = image.itemsize  # byte size of each value
    height, width = image.shape
    if stride is None:
        stride = window_size
    else:
        stride = int(stride)

    if height < window_size or width < window_size:
        return None

    new_height = 1 + (height - window_size) // stride
    new_width = 1 + (width - window_size) // stride

    shape = (new_height, new_width, window_size, window_size)
    strides = size * np.array([width * stride, stride, width, 1])
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    windows = windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2], windows.shape[3], 1)

    if enable_duplicate:
        extra_windows = []
        if (height - window_size) % stride != 0:
            for x in range(0, width - window_size, stride):
                extra_windows.append(image[height - window_size - 1:height - 1, x:x + window_size:])

        if (width - window_size) % stride != 0:
            for y in range(0, height - window_size, stride):
                extra_windows.append(image[y: y + window_size, width - window_size - 1:width - 1])

        if len(extra_windows) > 0:
            org_size = windows.shape[0]
            windows = np.resize(windows,
                                [org_size + len(extra_windows), windows.shape[1], windows.shape[2], windows.shape[3]])
            for i in range(len(extra_windows)):
                extra_windows[i] = extra_windows[i].reshape([extra_windows[i].shape[0], extra_windows[i].shape[1], 1])
                windows[org_size + i] = extra_windows[i]

    return windows


def xavier_cnn_initializer(shape, uniform=True):
    fan_in = shape[0] * shape[1] * shape[2]
    fan_out = shape[0] * shape[1] * shape[3]
    n = fan_in + fan_out
    if uniform:
        init_range = math.sqrt(6.0 / n)
        return tf.random_uniform(shape, minval=-init_range, maxval=init_range)
    else:
        stddev = math.sqrt(3.0 / n)
        return tf.truncated_normal(shape=shape, stddev=stddev)


def he_initializer(shape):
    n = shape[0] * shape[1] * shape[2]
    stddev = math.sqrt(2.0 / n)
    return tf.truncated_normal(shape=shape, stddev=stddev)


def weight(shape, stddev=0.01, name="weight", uniform=False, initializer="stddev"):
    if initializer == "xavier":
        initial = xavier_cnn_initializer(shape, uniform=uniform)
    elif initializer == "he":
        initial = he_initializer(shape)
    elif initializer == "uniform":
        initial = tf.random_uniform(shape, minval=-2.0 * stddev, maxval=2.0 * stddev)
    elif initializer == "stddev":
        initial = tf.truncated_normal(shape=shape, stddev=stddev)
    elif initializer == "identity":
        initial = he_initializer(shape)
        if len(shape) == 4:
            initial = initial.eval()
            i = shape[0] // 2
            j = shape[1] // 2
            for k in range(min(shape[2], shape[3])):
                initial[i][j][k][k] = 1.0
    else:
        initial = tf.zeros(shape)

    return tf.Variable(initial, name=name)


def bias(shape, initial_value=0.0, name=None):
    initial = tf.constant(initial_value, shape=shape)

    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)


# utilities for logging -----

def add_summaries(scope_name, model_name, var, save_stddev=True, save_mean=False, save_max=False, save_min=False):
    with tf.name_scope(scope_name):
        mean_var = tf.reduce_mean(var)
        if save_mean:
            tf.summary.scalar("mean/" + model_name, mean_var)

        if save_stddev:
            stddev_var = tf.sqrt(tf.reduce_mean(tf.square(var - mean_var)))
            tf.summary.scalar("stddev/" + model_name, stddev_var)

        if save_max:
            tf.summary.scalar("max/" + model_name, tf.reduce_max(var))

        if save_min:
            tf.summary.scalar("min/" + model_name, tf.reduce_min(var))
        tf.summary.histogram(model_name, var)


def get_now_date():
    d = datetime.datetime.today()
    return "%s/%s/%s %s:%s:%s" % (d.year, d.month, d.day, d.hour, d.minute, d.second)


def get_loss_image(image1, image2, scale=1.0, border_size=0):
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    if image1.dtype == np.uint8:
        image1 = image1.astype(np.double)
    if image2.dtype == np.uint8:
        image2 = image2.astype(np.double)

    loss_image = np.multiply(np.square(np.subtract(image1, image2)), scale)
    loss_image = np.minimum(loss_image, 255.0)
    loss_image = loss_image[border_size:-border_size, border_size:-border_size, :]

    return loss_image


def compute_mse(image1, image2, border_size=0):
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    if image1.dtype != np.uint8:
        image1 = image1.astype(np.int)
    image1 = image1.astype(np.double)

    if image2.dtype != np.uint8:
        image2 = image2.astype(np.int)
    image2 = image2.astype(np.double)

    mse = 0.0
    for i in range(border_size, image1.shape[0] - border_size):
        for j in range(border_size, image1.shape[1] - border_size):
            for k in range(image1.shape[2]):
                error = image1[i, j, k] - image2[i, j, k]
                mse += error * error

    return mse / ((image1.shape[0] - 2 * border_size) * (image1.shape[1] - 2 * border_size) * image1.shape[2])


def compute_psnr_and_ssim(image1, image2, border_size=0):
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    image1 = image1.astype(np.double)
    image2 = image2.astype(np.double)

    if border_size > 0:
        image1 = image1[border_size:-border_size, border_size:-border_size, :]
        image2 = image2[border_size:-border_size, border_size:-border_size, :]

    psnr = compare_psnr(image1, image2, data_range=255)
    ssim = compare_ssim(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                        sigma=1.5, data_range=255)
    return psnr, ssim


def print_filter_weights(tensor):
    print("Tensor[%s] shape=%s" % (tensor.name, str(tensor.get_shape())))
    weight = tensor.eval()
    for i in range(weight.shape[3]):
        values = ""
        for x in range(weight.shape[0]):
            for y in range(weight.shape[1]):
                for c in range(weight.shape[2]):
                    values += "%2.3f " % weight[y][x][c][i]
        print(values)
    print("\n")


def print_filter_biases(tensor):
    print("Tensor[%s] shape=%s" % (tensor.name, str(tensor.get_shape())))
    bias = tensor.eval()
    values = ""
    for i in range(bias.shape[0]):
        values += "%2.3f " % bias[i]
    print(values + "\n")


def get_psnr(mse, max_value=255.0):
    if mse is None or mse == float('Inf') or mse == 0:
        psnr = 0
    else:
        psnr = 20 * math.log(max_value / math.sqrt(mse), 10)
    return psnr


def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

    if output_to_logging:
        if output_detail:
            logging.info(parameters_string)
        logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))