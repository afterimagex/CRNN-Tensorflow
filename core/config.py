# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os.path as osp

from utils.attrdict import AttrDict as edict

__C = edict()
cfg = __C

###########################################
#                                         #
#            Training Options             #
#                                         #
###########################################

__C.TRAIN = edict()

# Initialize network with weights from this file
__C.TRAIN.WEIGHTS = ''

# Database to train
__C.TRAIN.DATABASE = ''

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (32,)

# Max pixel size of the longest side of a scaled input image
# A square will be used if value < 1
__C.TRAIN.MAX_SIZE = 256

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 12

__C.TRAIN.EPOCH = 100

__C.TRAIN.THREADS = 10

# Use shuffle when reading database
__C.TRAIN.USE_SHUFFLE = True

# Use data augment when reading database
__C.TRAIN.DATA_AUGMENT = True

###########################################
#                                         #
#              Model Options              #
#                                         #
###########################################


__C.MODEL = edict()

# The backbone
# ('mobilnet',)
__C.MODEL.BACKBONE = 'mobilnet'

# The number of classes in the dataset
__C.MODEL.NUM_CLASSES = -1

# Keep it for TaaS DataSet
__C.MODEL.CLASSES = []

###########################################
#                                         #
#             Solver Options              #
#                                         #
###########################################


__C.SOLVER = edict()

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.001

# End learning rate for the specified schedule
__C.SOLVER.END_LR = 0.00001

__C.SOLVER.POWER = 1.0

# Optional scaling factor for total loss
# This option is helpful to scale the magnitude
# of gradients during FP16 training
__C.SOLVER.LOSS_SCALING = 1.

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = 'exponential_decay'

__C.SOLVER.OPT_POLICY = 'adam'

# Hyperparameter used by the specified policy
# For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Uniform step size for 'steps' policy
__C.SOLVER.STEP_SIZE = 30000

__C.SOLVER.DECAY_RATE = 0.96

__C.SOLVER.USE_MOVING_AVERAGE_DECAY = True

__C.SOLVER.MOVING_AVERAGE_DECAY = 0.997

__C.SOLVER.STEPS = []

# Maximum number of SGD iterations
__C.SOLVER.MAX_ITERS = 40000

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyper parameters
__C.SOLVER.WEIGHT_DECAY = 0.0005

# L2 norm factor for clipping gradients
__C.SOLVER.CLIP_NORM = -1.0

# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_ITERS = 500

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0

# The steps for accumulating gradients
__C.SOLVER.ITER_SIZE = 1

# The interval to display logs
__C.SOLVER.DISPLAY = 20

# The interval to snapshot a model
__C.SOLVER.SNAPSHOT_ITERS = 5000

# prefix to yield the path: <prefix>_iters_XYZ.caffemodel
__C.SOLVER.SNAPSHOT_PREFIX = ''

###########################################
#                                         #
#               Test Options              #
#                                         #
###########################################

__C.TEST = edict()
__C.TEST.VIS = False
__C.TEST.BATCH_SIZE = 8
__C.TEST.THREADS = 10
__C.TEST.DATABASE = ''

###########################################
#                                         #
#               Misc Options              #
#                                         #
###########################################


# Number of GPUs to use (applies to both training and testing)
__C.NUM_GPUS = 1

# Use NCCL for all reduce, otherwise use cuda-aware mpi
__C.USE_NCCL = True

# Hosts for Inter-Machine communication
__C.HOSTS = []

__C.RESTORE = True

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
__C.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

# Default weights on (dx, dy, dw, dh, da) for normalizing rbox regression targets
__C.RBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0, 10.0)

# Prior prob for the positives at the beginning of training.
# This is used to set the bias init for the logits layer
__C.PRIOR_PROB = 0.01

# For reproducibility
__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Place outputs under an experiments directory
__C.EXP_DIR = ''

__C.CHARACTER_TXT = ''

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0

# Dump detection visualizations
__C.VIS = False
__C.VIS_ON_FILE = False

# Score threshold for visualization
__C.VIS_TH = 0.7

# Write summaries by tensor board
__C.ENABLE_TENSOR_BOARD = False


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if not isinstance(a, dict): return
    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))
        # the types must match, too
        v = _check_and_coerce_cfg_value_type(v, b[k], k)
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    global __C
    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value


def _check_and_coerce_cfg_value_type(value_a, value_b, key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b: return value_a
    if type_b is float and type_a is int: return float(value_a)

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_a, dict) and isinstance(value_b, edict):
        value_a = edict(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, key)
        )
    return value_a
