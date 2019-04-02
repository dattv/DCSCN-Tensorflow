"""
    Utility functions
"""

import time

import numpy as np
import os
import logging
import tensorflow as tf

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
    file_list = [path + f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and not f.startswith('.'))]
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

