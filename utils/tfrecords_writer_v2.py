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

__author__ = 'peichao.xu'

import io
import os
import time
import numpy as np
import tensorflow as tf
import utils.logger as logger
from multiprocessing import Queue, Process, Manager
from PIL import Image


class TextFeatureWriter(object):
    SENTINEL = ("", [])
    SAMPLE_QUEUE = Queue()

    def __init__(self):
        pass

    @staticmethod
    def int64_feature(value):
        ''' Wrapper for inserting int64 features into Example proto.'''
        value = value if isinstance(value, list) else [value]
        value = map(round, value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        ''' Wrapper for inserting float features into Example proto.'''
        value = value if isinstance(value, list) else [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def bytes_feature(value):
        ''' Wrapper for inserting bytes features into Example proto.'''
        value = value if isinstance(value, list) else [value]
        value = [val if isinstance(val, bytes) else val.encode('utf-8') for val in value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def char_count(label_files, split_char):
        logger.info("\n== Count Character ==")
        all_chars = []
        for label_file in label_files:
            with open(label_file, 'r', encoding='utf-8') as fin:
                for line in fin.readlines():
                    try:
                        filename, chars = line.strip().split(split_char)
                        for c in chars:
                            if c not in all_chars:
                                all_chars.append(c)
                    except ValueError as e:
                        logger.debug(e)
                        continue
        all_chars = sorted(list(set(all_chars)))
        logger.info('all_chars: {}'.format(all_chars))
        logger.info('num_class: {}'.format(len(all_chars)))
        return all_chars

    @staticmethod
    def txt_anno_to_data(imgs_dir, label_file, character):
        data = []
        with open(label_file, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                try:
                    img_name, text = line.strip().split('\t')
                except ValueError as e:
                    logger.debug(e)
                    continue
                data.append((os.path.join(imgs_dir, img_name), [character.encode(x) for x in text]))
        return data

    @staticmethod
    def splist(slist, each_split, num_split=None):
        if num_split:
            each_split = int(np.ceil(len(slist) / num_split))
        return [slist[i:i + each_split] for i in range(len(slist)) if i % each_split == 0]

    @staticmethod
    def image_check_and_convert(bytesbuf):
        if not bytesbuf.startswith(b'\xff\xd8'):
            return False, None

        elif bytesbuf[6:10] in (b'JFIF', b'Exif'):
            if not bytesbuf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                return False, None
        else:
            try:
                Image.open(io.BytesIO(bytesbuf)).verify()
            except Exception as e:
                logger.debug(e)
                return False, None

        return True, bytesbuf

    def data_to_queue(self, data, idx):
        for (impath, label) in data: self.SAMPLE_QUEUE.put((impath, label))
        logger.info(
            "Process-{} {:d} finished feed data to Queue. [{}]".format(idx, os.getpid(), time.strftime('%H:%M:%S')))

    def _write_serialize(self, tfrecords_path, num_eachs, idx):
        start = time.time()
        writer = tf.python_io.TFRecordWriter(tfrecords_path)
        for n in range(num_eachs):
            sample = self.SAMPLE_QUEUE.get()
            if sample == self.SENTINEL:
                self.SAMPLE_QUEUE.put(self.SENTINEL)
                break

            encoded = tf.gfile.GFile(sample[0], 'rb').read()
            flag, encoded = self.image_check_and_convert(encoded)
            if not flag: continue

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': self.bytes_feature(encoded),
                'label/encoded': self.int64_feature(sample[1]),
            })).SerializeToString()
            writer.write(example)

            if n % 100 == 0:
                print('Process-{} {:d} [{}/{}] [{}]'.format(i, os.getpid(), n, num_eachs, time.strftime('%H:%M:%S')))

        writer.close()
        end = time.time()
        print(
            'Process-{} {:d} finished writing Work. [{}] using time: {} s'.format(idx, os.getpid(),
                                                                                  time.strftime('%H:%M:%S'),
                                                                                  end - start))

    def write(self, data, save_dir, prefix, num_process=1, each_split=None):
        start = time.time()

        os.makedirs(save_dir, exist_ok=True)
        np.random.shuffle(data)

        logger.info("number of samples: {}, using {} process".format(len(data), num_process))

        multi_data = self.splist(data, None, num_process)

        num_files = num_process if each_split is None else int(np.ceil(1.0 * len(data) / each_split))
        num_eachs = int(np.ceil(1.0 * len(data) / num_process)) if each_split is None else each_split

        logger.info("number of files: {}, the prefix is \"{}\"".format(num_files, prefix))
        logger.info("each file are {} samples.".format(num_eachs))

        logger.info("== Loading Data To Queue.. ==")
        loader = []
        for i in range(num_process):
            p = Process(target=self.data_to_queue, args=(multi_data[i], i))
            p.start()
            loader.append(p)
        for p in loader:
            p.join()

        for i in range(num_process): self.SAMPLE_QUEUE.put(self.SENTINEL)

        print("== Begin Write TFRecords.. ==")
        writer = []
        for i in range(num_files):
            tfrecord_path = os.path.join(save_dir, '{:s}_{:05d}.tfrecords'.format(prefix, i))
            p = Process(target=self._write_serialize, args=(tfrecord_path, num_eachs, i))
            p.start()
            writer.append(p)

        for p in writer:
            # p.start()
            p.join()

        # writer = self.splist(writer, each_split=None, num_split=num_process)
        # for wp in writer:
        #     for p in wp:
        #         p.start()
        #     for p in wp:
        #         p.join()

        end = time.time()
        print("write tfrecords using time: {} s".format(end - start))


if __name__ == '__main__':
    from core.character import Character

    tfr = TextFeatureWriter()

    labels = tfr.char_count([r'D:\DATA\DRIVER\OCR数据\output\train.txt'], '\t')

    character = Character(labels)

    data = tfr.txt_anno_to_data(imgs_dir=r'D:\DATA\DRIVER\OCR数据\output\imgs',
                                label_file=r'D:\DATA\DRIVER\OCR数据\output\train.txt',
                                character=character)

    train_data = data[:10000]

    tfr.write(train_data, r'D:\DATA\DRIVER\OCR数据\output\tfrecords', 'train',
              num_process=10, each_split=1000)
