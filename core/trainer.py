# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/train.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from core.config import cfg
from utils import logger
from core.solver import Solver
from models import get_models
from utils.prettytable import PrettyTable

slim = tf.contrib.slim


class TrainWrapper(object):
    def __init__(self, coordinator):
        self.output_dir = coordinator.checkpoints_dir()
        self.summary_dir = coordinator.summary_dir()
        self.solver = Solver()

    def init_weights_fn(self, init_weights):
        if os.path.exists(init_weights):
            logger.info('Loading weights from {}.'.format(init_weights))
            exclusions = ['learning_rate', 'global_step', 'OptimizeLoss',
                          'crnn/LSTMLayers/fully_connected']
            variables = slim.get_variables_to_restore(exclude=exclusions)
            init_fn = slim.assign_from_checkpoint_fn(init_weights, variables, ignore_missing_vars=True)
            return init_fn
        else:
            raise ValueError('Invalid path of weights: {}'.format(init_weights))

    def tower_loss(self, logits, labels, seqlen):
        ctc_loss = tf.nn.ctc_loss(labels=labels,
                                  inputs=logits,
                                  sequence_length=seqlen,
                                  ignore_longer_outputs_than_inputs=True)
        ctc_loss_mean = tf.reduce_mean(ctc_loss)
        # total_loss = tf.add_n([ctc_loss_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.add_to_collection('losses', ctc_loss_mean)
        losses = tf.get_collection('losses')
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
        return ctc_loss_mean, total_loss

    def metrics(self, logits, labels, seqlen):
        softmax = tf.nn.softmax(logits, dim=-1, name='softmax')
        decoded, log_prob = tf.nn.ctc_greedy_decoder(softmax, seqlen)
        distance = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
        return distance, decoded

    def train(self, data_loader):
        with tf.device('/cpu:0'):
            input_images, input_labels, input_widths = data_loader.read_with_bucket_queue(
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_threads=cfg.TRAIN.THREADS,
                num_epochs=cfg.TRAIN.EPOCH,
                shuffle=cfg.TRAIN.USE_SHUFFLE)

            if cfg.NUM_GPUS > 1:
                images_sp = tf.split(input_images, cfg.NUM_GPUS)
                labels_sp = tf.sparse_split(sp_input=input_labels, num_split=cfg.NUM_GPUS, axis=0)
                widths_sp = tf.split(input_widths, cfg.NUM_GPUS)
            else:
                images_sp = [input_images]
                labels_sp = [input_labels]
                widths_sp = [input_widths]

            tower_grads = []
            tower_distance = []
            for i, host in enumerate(cfg.HOSTS):
                reuse = i > 0
                with tf.device('/gpu:%d' % host):
                    with tf.name_scope('model_%d' % host) as scope:
                        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                            logits = get_models(cfg.MODEL.BACKBONE)(cfg.MODEL.NUM_CLASSES).build(images_sp[i], True)

                        seqlen = tf.cast(tf.floor_div(widths_sp[i], 2), tf.int32, name='sequence_length')
                        model_loss, total_loss = self.tower_loss(logits, labels_sp[i], seqlen)
                        distance, _ = self.metrics(logits, labels_sp[i], seqlen)

                        if not reuse and cfg.ENABLE_TENSOR_BOARD:
                            tf.summary.image(name='InputImages', tensor=images_sp[i])
                            tf.summary.scalar(name='ModelLoss', tensor=model_loss)
                            tf.summary.scalar(name='TotalLoss', tensor=total_loss)
                            tf.summary.scalar(name='Distance', tensor=distance)

                        self.solver.bn_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                        grads = self.solver.optimizer.compute_gradients(total_loss)

                        tower_grads.append(grads)
                        tower_distance.append(distance)

            self.solver.apply_gradients(tower_grads)
            if cfg.SOLVER.USE_MOVING_AVERAGE_DECAY:
                self.solver.apply_moving_average()

            train_op = self.solver.get_train_op()
            distance_mean = tf.reduce_mean(tower_distance)

            if cfg.ENABLE_TENSOR_BOARD:
                summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(self.summary_dir, tf.get_default_graph())

        saver = tf.train.Saver(tf.global_variables())
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            if cfg.RESTORE:
                logger.info('continue training from previous checkpoint')
                ckpt = tf.train.latest_checkpoint(self.output_dir)
                logger.debug(ckpt)
                saver.restore(sess, ckpt)
            else:
                # Load the pre-trained weights
                if cfg.TRAIN.WEIGHTS:
                    self.init_weights_fn(cfg.TRAIN.WEIGHTS)(sess)

            start = time.time()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while not coord.should_stop():
                    _, step, lr, ml, tl, dt = sess.run(
                        [train_op, self.solver.global_step, self.solver.learning_rate, model_loss, total_loss,
                         distance_mean])

                    if np.isnan(tl):
                        logger.error('Loss diverged, stop training')
                        break

                    if step % cfg.SOLVER.DISPLAY == 0:
                        avg_time_per_step = (time.time() - start) / 10
                        avg_examples_per_second = (10 * cfg.TRAIN.BATCH_SIZE * cfg.NUM_GPUS) / (time.time() - start)
                        start = time.time()
                        tb = PrettyTable(
                            ['Step', 'LR', 'ModelLoss', 'TotalLoss', 'sec/step', 'exp/sec', 'Distance'])
                        tb.add_row(
                            ['{}/{}'.format(step, cfg.SOLVER.MAX_ITERS), '{:.3f}'.format(lr), '{:.3f}'.format(ml),
                             '{:.3f}'.format(tl),
                             '{:.3f}'.format(avg_time_per_step),
                             '{:.3f}'.format(avg_examples_per_second), '{:.3f}'.format(dt)])
                        print(tb)

                        if cfg.ENABLE_TENSOR_BOARD:
                            summary_str = sess.run([summary_op, ])
                            summary_writer.add_summary(summary_str[0], global_step=step)

                    if step % cfg.SOLVER.SNAPSHOT_ITERS == 0:
                        saver.save(sess, os.path.join(self.output_dir, 'model.ckpt'),
                                   global_step=self.solver.global_step)

            except tf.errors.OutOfRangeError:
                logger.error('Epochs Complete!')
            finally:
                coord.request_stop()
            coord.join(threads)


def train_net(coordinator, data_loader):
    tw = TrainWrapper(coordinator)
    tw.train(data_loader)
