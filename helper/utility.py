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

    def print(self):
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
    if not os.path.exists(directory):
        os.makedirs(directory)


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
