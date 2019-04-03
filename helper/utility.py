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

    def start(self, time_id):
        self.start_times[time_id] = time.time()

    def end(self, time_id):
        self.times[time_id] += time.time() - self.start_times[time_id]
        self.counts[time_id] += 1

    def print_(self):
        for i in range(self.timer_count):
            if self.counts[i] > 0:
                total = 0
                print("Average of {}: {:,}[ms]".format(i, self.times[i] * 1000 / self.counts[i]))
                total += self.times[i]
                print("Total of {}: {:,}".format(i, total))


class LoadError(Exception):
    def __init__(self, message):
        self.message = message


def remove_genetic(path, __func__):
    try:
        __func__(path)
    except OSError as error:
        print("Os error: {0}".format(error))


def clean_dir(path):
    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    for x in files:
        full_path = os.path.join(path, x)
        if os.path.isfile(full_path):
            f = os.remove
            remove_genetic(full_path, f)
        elif os.path.isdir(full_path):
            clean_dir(full_path)
            f = os.rmdir
            remove_genetic(full_path, f)


def make_dir(directory):
    """
        Create a folder
    :param directory:
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def delete_dir(directory):
    if os.path.exists(directory):
        clean_dir(directory)
        os.rmdir(directory)

def remove_dir(directory):
    if os.path.isdir(directory):
        clean_dir(directory)
        os.rmdir(directory)


def get_files_in_directory(path):
    if not path.endswith("/"):
        path = path + "/"
    file_list = [path + f for f in os.listdir(path) if
                 (os.path.isfile(os.path.join(path, f)) and not f.startswith('.'))]
    return file_list


def set_logging(filename, stream_log_level, file_log_level, tf_log_level):
    stream_log = logging.StreamHandler()
    stream_log.setLevel(stream_log_level)

    file_log = logging.FileHandler(filename=filename)
    file_log.setLevel(file_log_level)

    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(stream_log)
    logger.addHandler(file_log)
    logger.setLevel((min(stream_log_level, file_log_level)))

    tf.logging.set_verbosity(tf_log_level)


def save_image(filename, image, print_console=False):
    if len(image.shape) >= 3 and image.shape[2] == 1:
        image = image.reshape(image.shape[0], image.shape[1])

    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        make_dir(directory)

    image = misc.toimage(image, cmin=0, cmax=255)
    misc.imsave(filename, image)

    if print_console:
        print("Saved: [{}]".format(filename))


def save_image_data(filename, image):
    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        make_dir(directory)

    np.save(filename, image)
    print("Saved [{}]".format(filename))


def convert_rgb_to_y(image):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
    y_image = image.dot(xform.T) + 16.0

    return y_image


def convert_rgb_to_ycbcr(image):
    """

    :param image:
    :return:
    """
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    xform = np.array(
        [[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0],
         [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
         [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])

    ycbcr_image = image.dot(xform.T)
    ycbcr_image[:, :, 0] += 16.0
    ycbcr_image[:, :, [1, 2]] += 128.0

    return ycbcr_image


def convert_ycbcr_to_rgb(ycbcr_image):
    """

    :param ycbcr_image:
    :return:
    """
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - 16.0
    rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - 128.0
    xform = np.array(
        [[298.082 / 256.0, 0, 408.583 / 256.0],
         [298.082 / 256.0, -100.291 / 256.0, -208.120 / 256.0],
         [298.082 / 256.0, 516.412 / 256.0, 0]])
    rgb_image = rgb_image.dot(xform.T)

    return rgb_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image):
    if len(y_image.shape) <= 2:
        y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image[:, :, 0]
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image)


def set_image_alignment(image, alignment):
    alignment = int(alignment)

    width, height = image.shape[1], image.shape[0]
    width = (width // alignment) * alignment
    height = (height // alignment) * alignment

    if image.shape[1] != width or image.shape[0] != height:
        image = image[:height, :width]

    if len(image.shape) >= 3 and image.shape[2] >= 4:
        image = image[:, :, 0:3]

    return image


def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]

    new_width = width * scale
    new_height = height * scale

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
    elif len(image.shape) == 4 and image.shape[2] == 4:
        # the image may has an alpha chanel
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)

    return image


def load_image(filename, width=0, height=0, chanels=0, alignment=0, print_console=True):
    if not os.path.isfile(filename):
        raise LoadError("File not found [{}]".format(filename))

    try:
        image = np.atleast_3d(misc.imread(filename))

        if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
            raise LoadError("Attribute mismatch")
        if chanels != 0 and image.shape[2] != chanels:
            raise LoadError("Attributes mismatch")
        if alignment != 0 and ((width % alignment) != 0) or ((height % alignment) != 0):
            raise LoadError("Attribute mismatch")

        # if there is alpha plane, then cut it
        if len(image.shape) >= 3 and image.shape[2] >= 4:
            image = image[:, :, 0:3]

        if print_console:
            print("Load [{}]: {} x {} x {}".format(filename, image.shape[0], image.shape[1], image.shape[2]))
    except IndexError:
        print("IndexError: file: [{}], shape[{}]".format(filename, image.shape))
        return None

    return image


def load_image_data(filename, width=0, height=0, chanels=0, alignment=0, print_console=True):
    if not os.path.isfile(filename):
        raise LoadError("File not found")

    image = np.load(filename)

    if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
        raise LoadError("Attribute mismatch")
    if chanels != 0 and image.shape[2] != chanels:
        raise LoadError("Attribute mismatch")
    if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
        raise LoadError("Attribute mismatch")

    if print_console:
        print("Loaded [{}] {} x {} x {}".format(filename, image.shape[0], image.shape[1], image.shape[2]))
    return image


def get_split_images(image, window_size, stride=None, enable_duplicate=False):
    """
        Divide image with given stride and window_size
    :param image:
    :param window_size:
    :param stride:
    :param enable_duplicate:
    :return:
    """
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
    windows = np.lib.stride_tricks.as_strided(image, shape, strides=strides)
    windows = windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2], windows.shape[3], 1)

    if enable_duplicate:
        extra_windows = []
        if (height - window_size) % stride != 0:
            for x in range(0, width - window_size, stride):
                extra_windows.append((image[height - window_size - 1: height - 1, x: x + window_size]))

        if (width - window_size) % stride != 0:
            for y in range(0, height - window_size, stride):
                extra_windows.append(image[y: y + window_size, width - window_size - 1: width - 1])

        if len(extra_windows) > 0:
            org_size = windows.shape[0]
            windows = np.resize(windows,
                                [org_size + len(extra_windows), windows.shape[1], windows.shape[2],
                                 windows.shape[3]])
            for i in range(len(extra_windows)):
                extra_windows[i] = extra_windows[i].reshape(
                    [extra_windows[i].shape[0], extra_windows[i].shape[1], 1])
                windows[org_size + i] = extra_windows[i]

    return windows


def get_divided_images(image, window_size, stride, min_size=0):
    """
        Divide images with given stride. note return image size may not equal to window size.
    :param image:
    :param window_size:
    :param stride:
    :param min_size:
    :return:
    """
    h, w = image.shape[:2]
    divided_images = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            new_h = window_size if y + window_size <= h else h - y
            new_w = window_size if x + window_size <= w else w - x

            if new_h < min_size or new_w < min_size:
                continue
            divided_images.append(image[y: y + new_h, x: x + new_w, :])
    return divided_images


def xavier_cnn_initializer(shape, uniform=True):
    """
        initial values for tensor
    :param shape:
    :param uniform:
    :return:
    """
    fan_in = shape[0] * shape[1] * shape[2]
    fan_out = shape[0] * shape[1] * shape[3]

    n = fan_in + fan_out
    if uniform:
        init_range = math.sqrt(6.e0 / float(n))
        return tf.random_uniform(shape, minval=-init_range, maxval=init_range)
    else:
        stddev = math.sqrt(3.e0 / float(n))
        return tf.truncated_normal(shape=shape, stddev=stddev)


def he_initializer(shape):
    n = shape[0] * shape[1] * shape[2]
    stddev = math.sqrt(2.e0 / float(n))
    return tf.truncated_normal(shape=shape, stddev=stddev)


def upsample_filter(size):
    factor = (size + 1) // 2
    if size % 2 == 0:
        cneter = factor - 1
    else:
        center = factor - 0.5e0

    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def get_upscale_filter_size(scale):
    return 2 * scale - scale % 2


def upscale_weight(scale, chanels, name='weight'):
    cnn_size = get_upscale_filter_size(scale)

    initial = np.zeros(shape=[cnn_size, cnn_size, chanels, chanels], dtype=np.float32)
    filter_matrix = upsample_filter(cnn_size)

    for i in range(chanels):
        initial[:, :, i, i] = filter_matrix

    return tf.Variable(initial, name=name)


def weight(shape, stddev=0.01, name="weight", uniform=False, initializer="stddev"):
    if initializer == "xavier":
        initial = xavier_cnn_initializer(shape, uniform=uniform)

    elif initializer == "he":
        initial = he_initializer(shape)

    elif initializer == "uniform":
        initial = tf.random_uniform(shape, minval=-2. * stddev, maxval=2. * stddev)

    elif initializer == "stddev":
        initial = tf.truncated_normal(shape, stddev=stddev)

    elif initializer == "identity":
        initial = he_initializer(shape)
        if len(shape) == 4:
            initial = initial.eval()
            i = shape[0] // 2
            j = shape[1] // 2
            for k in range(min(shape[2], shape[3])):
                initial[i][j][k][k] = 1.e0
    else:
        initial = tf.zeros(shape)

    return tf.Variable(initial, name=name)

def bias(shape, initial_value=0.e0, name=None):
    initial = tf.constant(initial_value, shape=shape)

    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)


# utilities for logging --------------

def get_shapes(input_tensor):
    return input_tensor.get_shape().as_list()


def add_summaries(scope_name, model_name, var, header_name="", save_stddev=True, save_mean=False, save_max=False,
                  save_min=False):
    with tf.name_scope(scope_name):
        mean_var = tf.reduce_mean(var)

        if save_mean:
            tf.summary.scalar(header_name + "mean/" + model_name, mean_var)

        if save_stddev:
            stddev_var = tf.sqrt(tf.reduce_mean(tf.square(var - mean_var)))
            tf.summary.scalar(header_name + "stddev/" + model_name, stddev_var)

        if save_max:
            tf.summary.scalar(header_name + "max/" + model_name, tf.reduce_max(var))

        if save_min:
            tf.summary.scalar(header_name, "min/" + model_name, tf.reduce_min(var))

        tf.summary.histogram(header_name + model_name, var)


def log_scalar_value(writer, name, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
    writer.add_summary(summary, step)


def log_fcn_output_as_image(image, width, height, filters, model_name, max_outputs=20):
    """
        input tensor should be [W, H, In_Ch, Out_Ch]
        so transform to [ In_Ch * Out_Ch, W, H] and visualize if
    :param image:
    :param width:
    :param height:
    :param filters:
    :param model_name:
    :param max_outputs:
    :return:
    """
    reshaped_image = tf.reshape(image, [-1, height, width, filters])
    tf.summary.image(model_name, reshaped_image[:, :, :, :1], max_outputs=max_outputs)


def log_cnn_weights_as_image(model_name, weights, max_outputs=20):
    """
    input tensor shourld be [W, H, In_Ch, Out_ch]
    so transform to [In_ch * Out_ch, W, H]
    :param model_name:
    :param weights:
    :param max_outputs:
    :return:
    """
    shapes = get_shapes(weights)
    weights = tf.reshape(weights, [shapes[0], shapes[1], shapes[2] * shapes[3]])
    weights_transposed = tf.transpose(weights, [2, 0, 1])
    weights_transposed = tf.reshape(weights_transposed, [shapes[2] * shapes[3], shapes[0], shapes[1], 1])
    tf.summary.image(model_name, weights_transposed, max_outputs=max_outputs)


def get_now_date():
    d = datetime.datetime.today()
    return "%s/%s/%s %s:%s:%s" % (d.year, d.month, d.day, d.hour, d.minute, d.second)


def get_loss_image(image1, image2, scale=1.e0, border_size=0):
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    loss_image = np.multiply(np.square(np.subtract(image1, image2)), scale)
    loss_image = np.minimum(loss_image, 255.e0)

    if border_size > 0:
        loss_image = loss_image[border_size:-border_size, border_size:-border_size, :]

    return loss_image


def trim_image_as_file(image):
    image = np.rint(image)
    image = np.clip(image, 0, 255)

    if image.dtype != np.float32:
        image = image.astype(np.float32)

    return image


def compute_PSNR_and_SSIM(image1, image2, border_size=0):
    """
    Compute PSNR and SSIM index form 2 images
    we round it and clib to 0 - 255. then shape "scale pixesl from each border
    :param image1:
    :param image2:
    :param border_size:
    :return:
    """
    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)

    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
        return None

    image1 = trim_image_as_file(image1)
    image2 = trim_image_as_file(image2)

    if border_size > 0:
        image1 = image1[border_size:-border_size, border_size:-border_size, :]
        image2 = image2[border_size:-border_size, border_size:-border_size, :]

    PSNR = compare_psnr(image1, image2, data_range=255)
    SSIM = compare_ssim(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                        sigma=1.5, data_range=255)

    return PSNR, SSIM


def print_filter_weights(tensor):
    print("Tensor [{}] shape={}".format(tensor.name, str(tensor.get_shape())))

    weight_value = tensor.eval()
    for i in range(weight_value.shape[3]):
        values = ""
        for x in range(weight_value.shape[0]):
            for y in range(weight_value.shape[1]):
                for c in range(weight_value.shape[2]):
                    values += "%2.3f" % weight_value[y][x][c][i]
        print(values)
    print("\n")


def print_filter_bias(tensor):
    print("Tensor[%s] shape=%s" % (tensor.name, str(tensor.get_shape())))
    bias = tensor.eval()
    values = ""
    for i in range(bias.shape[0]):
        values += "%2.3f " % bias[i]

    print(values + "\n")


def get_PSNR(mse, max_value=255.e0):
    if mse is None or mse == float("Inf") or mse == 0:
        PSNR = 0
    else:
        PSNR = 20.e0 * math.log(max_value / math.sqrt(mse), 10.e0)
    return PSNR


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


def flip(image, flip_type, invert=False):
    if flip_type == 0:
        return image
    elif flip_type == 1:
        return np.flipud(image)
    elif flip_type == 2:
        return np.fliplr(image)
    elif flip_type == 3:
        return np.flipud(np.fliplr(image))
    elif flip_type == 4:
        return np.rot90(image, 1 if invert is False else -1)
    elif flip_type == 5:
        return np.rot90(image, -1 if invert is False else 1)
    elif flip_type == 6:
        if invert is False:
            return np.flipud(np.rot90(image))
        else:
            return np.rot90(np.flipud(image), -1)
    elif flip_type == 7:
        if invert is False:
            return np.flipud(np.rot90(image, -1))
        else:
            return np.rot90(np.flipud(image), 1)
