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

import tensorflow as tf
from core.config import cfg
from models import get_models
from utils.prettytable import PrettyTable
from core.character import Character


class TestServer(object):
    def __init__(self, coordinator):
        self.output_dir = coordinator.checkpoints_dir()
        self.decoder = Character().from_txt(cfg.CHARACTER_TXT)

    def evaluate(self, data_loader):
        with tf.device('/cpu:0'):
            input_images, input_labels, input_widths = data_loader.read_with_bucket_queue(
                batch_size=cfg.TEST.BATCH_SIZE,
                num_threads=cfg.TEST.THREADS,
                num_epochs=1,
                shuffle=False)
            with tf.device('/gpu:0'):
                logits = get_models(cfg.MODEL.BACKBONE)(cfg.MODEL.NUM_CLASSES).build(input_images, False)
                seqlen = tf.cast(tf.floor_div(input_widths, 2), tf.int32, name='sequence_length')

                softmax = tf.nn.softmax(logits, dim=-1, name='softmax')
                decoded, log_prob = tf.nn.ctc_greedy_decoder(softmax, seqlen)
                distance = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

        saver = tf.train.Saver(tf.global_variables())
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.output_dir))
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                cnt = 0
                dm = 1e-5
                while not coord.should_stop():
                    dt = sess.run([distance, ])[0]

                    cnt += 1
                    dm = (dm + dt) / cnt

                    if cfg.TEST.VIS:
                        dd, il, ii = sess.run([decoded, input_labels, input_images])

                        gts = self.decoder.sparse_to_strlist(il.indices, il.values, cfg.TEST.BATCH_SIZE)
                        pts = self.decoder.sparse_to_strlist(dd[0].indices, dd[0].values, cfg.TEST.BATCH_SIZE)

                        tb = PrettyTable()
                        tb.field_names = ['Index', 'GroundTruth', 'Predict', '{:.3f}/{:.3f}'.format(dt, dm)]
                        for i in range(len(gts)):
                            tb.add_row([i, gts[i], pts[i], ''])
                        print(tb)
                    else:
                        print('EditDistance: {:.3f}/{:.3f}'.format(dt, dm))

            except tf.errors.OutOfRangeError:
                print('Epochs Complete!')
            finally:
                coord.request_stop()
            coord.join(threads)


class InferServer(object):
    def __init__(self, coordinator, data_loader):
        self.images_dir = cfg.TEST.DATABASE
