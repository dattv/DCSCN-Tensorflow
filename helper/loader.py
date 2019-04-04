"""
    Function for loading converting data
"""

from helper import utility as util
import os
import configparser
import numpy as np
from scipy import misc
import random

INPUT_IMAGE_DIR = "input"
INTERPOLATED_IMAGE_DIR = "interpolated"
TRUE_IMAGE_DIR = "true"


def build_image_set(file_path, chanels=1, scale=1, convert_ycbcr=True, resampleing_method="bicubic",
                    print_console=True):
    true_image = util.set_image_alignment(util.load_image(file_path, print_console=print_console), scale)

    if chanels == 1 and true_image.shape[2] == 3 and convert_ycbcr:
        true_image = util.convert_rgb_to_y(true_image)

    input_image = util.resize_image_by_pil(true_image, 1.e0 / scale, resampleing_method=resampleing_method)
    input_interpolated_image = util.resize_image_by_pil(input_image, scale, resampleing_method=resampleing_method)

    return input_image, input_interpolated_image, true_image

def build_input_image(image, width=0, height=0, chanels=1, scale=1, alignment=0, convert_ycbcr=True):
    """
    build input image from file,
    crop, adjust the image alignment for the scale factor, resize, conver color image
    :param image:
    :param width:
    :param height:
    :param chanels:
    :param scale:
    :param alignment:
    :param convert_ycbcr:
    :return:
    """
    if width != 0 and height != 0:
        if image.shape[0] != height or image.shape[1] != width:
            x = (image.shape[1] - width) // 2
            y = (image.shape[0] - height) // 2

            image = image[x: x + width, y: y + height, :]

    if alignment > 1:
        image = util.set_image_alignment(image, alignment)

    if chanels == 1 and image.shape[2] == 3:
        if convert_ycbcr:
            image = util.convert_rgb_to_y(image)

    else:
        if convert_ycbcr:
            image = util.convert_rgb_to_ycbcr(image)

    if scale != 1:
        image = util.resize_image_by_pil(image, 1.e0/scale)

    return image

def load_input_image(filename, width=0, height=0, chanels=1, scale=1, alignment=0, convert_ycbcr=True,
                     print_console=False):
    image = util.load_image(filename, print_console=print_console)
    return build_input_image(image, width, height, chanels, scale, alignment, convert_ycbcr)

class BatchDataSets:
    def __init__(self, scale, batch_dir, batch_image_size, stride_size=0, chanels=1, resampling_method="bicubic"):

        self.scale = scale
        self.batch_image_size = batch_image_size
        if stride_size == 0:
            self.stride = batch_image_size // 2

        else:
            self.stride = stride_size

        self.chanels = chanels
        self.resampling_method = resampling_method
        self.count = 0
        self.batch_dir = batch_dir
        self.batch_index = None

    def build_batch(self, data_dir):
        """
        Build batch images
        :param data_dir:
        :return:
        """
        print("Building batch images for {}...".format(self.batch_dir))
        filenames = util.get_files_in_directory(data_dir)

        images_count = 0

        util.make_dir(self.batch_dir)
        util.clean_dir(self.batch_dir)
        util.make_dir(self.batch_dir + "/" + INPUT_IMAGE_DIR)
        util.make_dir(self.batch_dir + "/" + INTERPOLATED_IMAGE_DIR)
        util.make_dir(self.batch_dir + "/" + TRUE_IMAGE_DIR)

        processed_images = 0

        for filename in filenames:
            output_window_size = self.batch_image_size * self.scale
            output_window_stride = self.stride * self.scale

            input_image, input_interpolated_image, true_image = build_image_set(filename, chanels=self.chanels,
                                                                                resampleing_method=self.resampling_method,
                                                                                scale=self.scale,
                                                                                print_console=False)

            # split into batch images
            input_batch_images = util.get_split_images(input_image, self.batch_image_size, stride=self.stride)
            input_interpolated_batch_images = util.get_split_images(input_interpolated_image,
                                                                    output_window_size,
                                                                    stride=self.stride)

            if input_batch_images is None or input_interpolated_batch_images is None:
                continue
            input_count = input_batch_images.shape[0]

            true_batch_image = util.get_split_images(true_image, output_window_size, stride=output_window_stride)

            # for i in range(input_count):
            #     self.save_input_batch_image()
            print("djkfldkj")

    def load_batch_counts(self):
        """
        load already built batch images.
        :return:
        """
        if not os.path.isdir(self.batch_dir):
            self.count = 0
            return

        config = configparser.ConfigParser()
        try:
            with open(self.batch_dir + "/batch_images.ini") as f:
                config.read_file(f)
            self.count = config.getint("batch", "count")

        except IOError:
            self.count = 0
            return

    def load_all_batch_images(self):
        print("Allocating memory for all batch images.")
        self.input_images = np.zeros(shape=[self.count, self.batch_image_size, self.batch_image_size, 1],
                                     dtype=np.uint8) # type np.ndarray

        self.input_interpolated_images = np.zeros(
            shape=[self.count, self.batch_image_size * self.scale, self.batch_image_size * self.scale, 1],
            dtype=np.uint8) # type np.ndarray

        self.true_images = np.zeros(
            shape=[self.count, self.batch_image_size * self.scale, self.batch_image_size * self.scale, 1],
            dtype=np.uint8) # type np.ndarray

        print("Loading all batch images.")
        for i in range(self.count):
            self.input_images[i] = self.load_input_batch_image(i)
            self.input_interpolated_images[i] = self.load_interpolated_batch_image(i)
            self.true_images[i] = self.load_true_batch_image(i)
            if i % 1000 == 0:
                print(".", end="", flush=True)
        print("Load finished.")

    def release_batch_images(self):
        if hasattr(self, "input_images"):
            del self.input_images
        self.input_images = None

        if hasattr(self, "input_interpolated_images"):
            del self.input_interpolated_images
        self.input_interpolated_images = None

        if hasattr(self, "true_images"):
            del self.true_images
        self.true_images = None

    def is_batch_exist(self):
        if not os.path.isdir(self.batch_dir):
            return False

        config = configparser.ConfigParser()

        try:
            with open(self.batch_dir + "/batch_images.ini") as f:
                config.read_file(f)

            if config.getint("batch", "count") <= 0:
                return False

            if config.getint("batch", "scale") != self.scale:
                return False

            if config.getint("batch", "batch_image_size") != self.batch_image_size:
                return False

            if config.getint("batch", "stride") != self.stride:
                return False

            if config.getint("batch", "chanels") != self.chanels:
                return False

            return True

        except IOError:
            return False

    def init_batch_index(self):
        self.batch_index = random.sample(range(0, self.count), self.count)
        self.index = 0

    def get_next_image_no(self):
        if self.index >= self.count:
            self.init_batch_index()

        image_no = self.batch_index[self.index]
        self.index += 1
        return image_no

    def load_batch_image_from_disk(self, image_number):
        image_number = image_number % self.count

        input_image = self.load_input_batch_image(image_number)
        input_interpolated = self.load_interpolated_batch_image(image_number)
        true = self.load_true_batch_image(image_number)

        return input_image, input_interpolated, True

    def load_input_batch_image(self, image_number):
        image = misc.imread(self.batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.bmp" % image_number)
        return image.reshape(image.shape[0], image.shape[1], 1)

    def load_interpolated_batch_image(self, image_number):
        image = misc.imread(self.batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.bmp" % image_number)
        return image.reshape(image.shape[0], image.shape[1], 1)

    def load_true_batch_image(self, image_number):
        image = misc.imread(self.batch_dir + "/" + TRUE_IMAGE_DIR, "/%06d.bmp" % image_number)
        return image.reshape(image.shape[0], image.shape[1], 1)

    def save_input_batch_image(self, image_number, image):
        return util.save_image(self.batch_dir + "/" + INPUT_IMAGE_DIR + "/%06d.bmp" % image_number, image)

    def save_interpolated_batch_image(self, image_number, image):
        return util.save_image(self.batch_dir + "/" + INTERPOLATED_IMAGE_DIR + "/%06d.bmp" % image_number, image)

    def save_true_batch_image(self, image_number, image):
        return util.save_image(self.batch_dir + "/" + TRUE_IMAGE_DIR + "/%06d.bmp" % image_number, image)