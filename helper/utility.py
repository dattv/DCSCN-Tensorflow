"""
    Utility functions
"""

import datetime
import logging
import math
import os
import time
from os import listdir

import numpy as np
import tensorflow as tf
from PIL import Image
from os.path import isfile, join
from scipy import misc

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
