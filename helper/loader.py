"""
    Function for loading converting data
"""

from helper import utility as util

INPUT_IMAGE_IDR = "input"
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
        util.make_dir(self.batch_dir + "/" + INPUT_IMAGE_IDR)
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