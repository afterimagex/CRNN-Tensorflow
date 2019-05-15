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

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

import sys
import pprint
import argparse
import utils.logger as logger

from core.config import cfg
from core.inference import TestServer
from core.coordinator import Coordinator
from utils.tfrecords_reader import TextFeatureReader


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Detection Network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--exp_dir', dest='exp_dir',
                        help='experiment dir',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    logger.info('Called with args:\n' + str(args))

    coordinator = Coordinator(args.cfg_file, args.exp_dir)

    logger.info('Using config:\n' + pprint.pformat(cfg))

    data_loader = TextFeatureReader(tfrecords_dir=cfg.TEST.DATABASE,
                                    prefix='test',
                                    augment_online=False)

    test_server = TestServer(coordinator)
    test_server.evaluate(data_loader)
