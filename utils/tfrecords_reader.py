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
import cv2
import glob
import numpy as np
import tensorflow as tf

from utils.data_augment import dataAugment


class TextFeatureReader(object):
    def __init__(self, tfrecords_dir, prefix, **kwargs):
        super(TextFeatureReader, self).__init__()

        self._tfrecords_files_path = glob.glob('{:s}/{:s}*.tfrecords'.format(tfrecords_dir, prefix))
        self._data_augment = kwargs.get('augment_online', False)

        self._features = {
            'image/encoded': tf.FixedLenFeature((), tf.string),
            'label/encoded': tf.VarLenFeature(tf.int64),
        }

    def __len__(self):
        counts = 0
        for record in self._tfrecords_files_path:
            counts += sum(1 for _ in tf.python_io.tf_record_iterator(record))
        return counts

    def _parse_single_example(self, serialized_example):
        features = tf.parse_single_example(serialized_example, features=self._features)
        image = tf.image.decode_image(features['image/encoded'], channels=3)
        if self._data_augment: image = dataAugment(image)
        image, width = tf.py_func(self._py_resize_image_and_pad, inp=[image],
                                  Tout=[tf.uint8, tf.int32])
        label = tf.cast(features['label/encoded'], tf.int32)
        return image, label, width

    def read_with_bucket_queue(self,
                               batch_size: int,
                               num_threads=10,
                               boundaries=[i * 32 for i in range(1, 32)],
                               num_epochs=None,
                               shuffle=True):
        reader = tf.TFRecordReader()
        capacity = num_threads * batch_size * 2
        data_queue = tf.train.string_input_producer(self._tfrecords_files_path,
                                                    capacity=capacity, num_epochs=num_epochs, shuffle=shuffle)
        _, serialized_example = reader.read(data_queue)
        image, label, width = self._parse_single_example(serialized_example)
        image.set_shape([32, None, 3])
        width.set_shape([])
        _, data_tuple = tf.contrib.training.bucket_by_sequence_length(
            input_length=tf.shape(image)[1],
            tensors=[image, label, width],
            bucket_boundaries=boundaries,
            batch_size=batch_size,
            capacity=capacity,
            allow_smaller_final_batch=False,
            dynamic_pad=True)
        images, labels, widths = data_tuple
        return images, labels, widths  # width shape is [batch,]

    @staticmethod
    def _py_resize_image_and_pad(image):
        nw = int(np.ceil(32.0 * image.shape[1] / image.shape[0]))
        nw32 = nw if nw % 32 == 0 else (nw // 32 + 1) * 32
        tmp = cv2.resize(image, (nw, 32))
        img = np.zeros((32, nw32, 3), dtype=np.uint8)
        img[:32, : nw, :] = tmp
        return img, np.int32(nw)

    @staticmethod
    def file_filters(fdir, patterns, endswith):
        return [os.path.join(fdir, f) for f in os.listdir(fdir) if f.endswith(endswith) and patterns in f]


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    tfr = TextFeatureReader(tfrecords_dir='D:/DATA/tftest',
                            prefix='train',
                            augment_online=False)

    batch_images, batch_labels, batch_widths = tfr.read_with_bucket_queue(batch_size=8,
                                                                          num_threads=10,
                                                                          num_epochs=10,
                                                                          shuffle=True)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                images, labels, widths = sess.run([batch_images, batch_labels, batch_widths])
                big = np.vstack(images).astype(np.uint8)
                plt.imshow(big)
                plt.show()

        except tf.errors.OutOfRangeError:
            print('Epochs Complete!')
        finally:
            coord.request_stop()
            coord.join(threads)
