"""
    Function for sharing arguments and their default values
    Reference: https://github.com/jiny2001/dcscn-super-resolution/blob/master/helper/args.py
"""

import sys
import numpy as np
import tensorflow as tf
from numpy.f2py.auxfuncs import F2PYError

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model network parameters
flags.DEFINE_integer("scale", 2, "Scale factor for Super Resolution (should be 2 or more)")
flags.DEFINE_integer("layers", 12, "Number of layers of feature xtraction CNNs")


def get():
    print("Python Interpreter version:%s" % sys.version[:3])
    print("Tensorflow version:$s" % tf.__version__)
    print("Numpy version:%s" % np.__version__)

    # check which library you are using
    return FLAGS
