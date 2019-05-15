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

import os
import io
import sys
import time
import glob
import numpy as np
import tensorflow as tf

from PIL import Image
from multiprocessing import Manager, Pool

if os.name == 'posix':
    SENTINEL = ("", [])
    SAMPLE_QUEUE = Manager().Queue()


def split_list(xlist, each_split, num_split=None):
    if num_split:
        each_split = int(np.ceil(len(xlist) / num_split))
    return [xlist[i:i + each_split] for i in range(len(xlist)) if i % each_split == 0]


def int64_feature(value):
    ''' Wrapper for inserting int64 features into Example proto.'''
    value = value if isinstance(value, list) else [value]
    value = map(round, value)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    ''' Wrapper for inserting float features into Example proto.'''
    value = value if isinstance(value, list) else [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    ''' Wrapper for inserting bytes features into Example proto.'''
    value = value if isinstance(value, list) else [value]
    value = [val if isinstance(val, bytes) else val.encode('utf-8') for val in value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def txt_anno_to_data(imgs_dir, label_file, character):
    data = []
    with open(label_file, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            try:
                img_name, text = line.strip().split('\t')
            except ValueError as e:
                print(e)
                continue
            data.append((os.path.join(imgs_dir, img_name), [character.encode(x) for x in text]))
    return data


def char_count(label_files, split_char):
    print("\n== Count Character ==")
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
                    print(e)
                    continue
    all_chars = sorted(list(set(all_chars)))
    print('all_chars:  ', all_chars)
    print('num_class:  ', len(all_chars))
    return all_chars


def tfrecords_write(data, save_dir, prefix, each_split=None):
    start = time.time()

    os.makedirs(save_dir, exist_ok=True)
    np.random.shuffle(data)
    print("\nnumber of samples: {}".format(len(data)))

    if each_split:
        multi_data = split_list(data, each_split)
    else:
        multi_data = [data]

    for i, mdata in enumerate(multi_data):
        tfrecord_path = os.path.join(save_dir, '{:s}_{:05d}.tfrecords'.format(prefix, i))
        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            count = 0
            all_count = len(mdata)
            for sample in mdata:
                encoded = tf.gfile.GFile(sample[0], 'rb').read()
                flag, encoded = image_check_and_convert(encoded)
                if not flag: continue
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': bytes_feature(encoded),
                    'label/encoded': int64_feature(sample[1]),
                })).SerializeToString()
                writer.write(example)
                sys.stdout.write(
                    '\r>> Writing {:d}/{:d} {:s} tfrecords'.format(count + 1, all_count, os.path.split(sample[0])[-1]))
                sys.stdout.flush()
                count += 1
            sys.stdout.write('\n')
            sys.stdout.flush()
    end = time.time()
    print("write tfrecords using time: {} s".format(end - start))


def sample_count(tfrecords_dir, prefix):
    tfrecords_file_paths = glob.glob('{:s}/{:s}*.tfrecords'.format(tfrecords_dir, prefix))

    counts = 0
    for record in tfrecords_file_paths:
        counts += sum(1 for _ in tf.python_io.tf_record_iterator(record))

    return counts


def data_to_queue(data, i):
    for (impath, label) in data: SAMPLE_QUEUE.put((impath, label))
    print("Process-{} {:d} finished feed data to Queue. [{}]".format(i, os.getpid(), time.strftime('%H:%M:%S')))


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
            print(e)
            return False, None

    return True, bytesbuf


def _write_serialize(tfrecords_path, num_eachs, i):
    start = time.time()
    writer = tf.python_io.TFRecordWriter(tfrecords_path)
    for n in range(num_eachs):
        sample = SAMPLE_QUEUE.get()
        if sample == SENTINEL:
            SAMPLE_QUEUE.put(SENTINEL)
            break

        encoded = tf.gfile.GFile(sample[0], 'rb').read()
        flag, encoded = image_check_and_convert(encoded)
        if not flag: continue

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(encoded),
            'label/encoded': int64_feature(sample[1]),
        })).SerializeToString()
        writer.write(example)

        if n % 100 == 0:
            print('Process-{} {:d} [{}/{}] [{}]'.format(i, os.getpid(), n, num_eachs, time.strftime('%H:%M:%S')))

    writer.close()
    end = time.time()
    print(
        'Process-{} {:d} finished writing Work. [{}] using time: {} s'.format(i, os.getpid(), time.strftime('%H:%M:%S'),
                                                                              end - start))


def multi_tfrecords_write(data, save_dir, prefix, num_process=10, each_split=None):
    start = time.time()

    os.makedirs(save_dir, exist_ok=True)
    np.random.shuffle(data)

    print("\nnumber of samples: {}, using {} process".format(len(data), num_process))

    multi_data = split_list(data, None, num_process)

    num_files = num_process if each_split is None else int(np.ceil(1.0 * len(data) / each_split))
    num_eachs = int(np.ceil(1.0 * len(data) / num_process)) if each_split is None else each_split

    print("number of files: {}, the prefix is \"{}\"".format(num_files, prefix))
    print("each file are {} samples.".format(num_eachs))

    print("\n ==Loading Data To Queue.. ==")
    inp = Pool(num_process)
    for i in range(num_process):
        inp.apply_async(data_to_queue, (multi_data[i], i))
    inp.close()
    inp.join()
    for i in range(num_process): SAMPLE_QUEUE.put(SENTINEL)

    print("\n== Begin Write TFRecords.. ==")
    outp = Pool(num_process)
    for i in range(num_files):
        tfrecord_path = os.path.join(save_dir, '{:s}_{:05d}.tfrecords'.format('train', i))
        outp.apply_async(_write_serialize, (tfrecord_path, num_eachs, i))

    outp.close()
    outp.join()

    end = time.time()
    print("write tfrecords using time: {} s".format(end - start))


if __name__ == '__main__':
    from core.character import Character

    # TODO: windows platform has some bug
    if os.name == 'nt':
        SENTINEL = ("", [])
        SAMPLE_QUEUE = Manager().Queue()

    labels = char_count(['/code/disk1/xupeichao/data/OCR/wecrash/rec/sim/0806_trainval.txt'], '\t')

    character = Character(labels)

    data = txt_anno_to_data(imgs_dir='/code/disk1/xupeichao/data/OCR/wecrash/rec/sim/imgs_0806',
                            label_file='/code/disk1/xupeichao/data/OCR/wecrash/rec/sim/0806_trainval.txt',
                            character=character)

    # tfrecords_write(data, '/code/disk1/xupeichao/data/OCR/wecrash/rec/sim/tftest', 'train')
    multi_tfrecords_write(data, '/code/disk1/xupeichao/data/OCR/wecrash/rec/sim/tftest', 'train',
                          num_process=10, each_split=2000)

    num_samples = sample_count('/code/disk1/xupeichao/data/OCR/wecrash/rec/sim/tftest', 'train')
    print("check record all samples: ", num_samples)
