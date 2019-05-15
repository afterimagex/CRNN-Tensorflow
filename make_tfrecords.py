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
from utils.tfrecords_writer import (
    tfrecords_write,
    multi_tfrecords_write,
    char_count,
    txt_anno_to_data,
)

if __name__ == '__main__':
    from core.character import Character

    if os.name == 'nt':
        from multiprocessing import Manager
        SENTINEL = ("", [])
        SAMPLE_QUEUE = Manager().Queue()

    labels = char_count([r'D:\DATA\DRIVER\OCR数据\output\train.txt'], '\t')

    character = Character(labels)

    data = txt_anno_to_data(imgs_dir=r'D:\DATA\DRIVER\OCR数据\output\imgs',
                            label_file=r'D:\DATA\DRIVER\OCR数据\output\train.txt',
                            character=character)

    train_sp = data[:10000]
    test_sp = data[10000:12000]

    multi_tfrecords_write(train_sp, r'D:\DATA\DRIVER\OCR数据\output\imgs\tfrecords', 'train',
                          num_process=10, each_split=1000)

    tfrecords_write(test_sp, r'D:\DATA\DRIVER\OCR数据\output\imgs\tfrecords', 'test')
