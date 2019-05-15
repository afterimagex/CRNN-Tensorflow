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
from utils import logger


class Solver(object):
    def __init__(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.set_learning_rate()
        self.set_optimizer()

        self.bn_updates_op = None
        self.apply_gradient_op = None
        self.variables_averages_op = tf.no_op(name='moving_average')

    def set_learning_rate(self):
        policy = cfg.SOLVER.LR_POLICY
        if policy == 'exponential_decay':
            self.learning_rate = tf.train.exponential_decay(learning_rate=cfg.SOLVER.BASE_LR,
                                                            global_step=tf.train.get_global_step(),
                                                            decay_steps=cfg.SOLVER.STEP_SIZE,
                                                            decay_rate=cfg.SOLVER.DECAY_RATE,
                                                            staircase=True,
                                                            name='learning_rate')
        elif policy == 'polynomial_decay':
            self.learning_rate = tf.train.polynomial_decay(learning_rate=cfg.SOLVER.BASE_LR,
                                                           global_step=tf.train.get_global_step(),
                                                           decay_steps=cfg.SOLVER.MAX_ITERS,
                                                           end_learning_rate=cfg.SOLVER.END_LR,
                                                           power=cfg.SOLVER.POWER,
                                                           name='learning_rate')
        else:
            raise ValueError('Unknown learning rate policy: ' + policy)

    def set_optimizer(self):
        policy = cfg.SOLVER.OPT_POLICY
        if policy == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        elif policy == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                        momentum=cfg.SOLVER.MOMENTUM)
        else:
            raise ValueError('Unknown optimizer policy: ' + policy)

    def apply_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        self.apply_gradient_op = self.optimizer.apply_gradients(average_grads, global_step=self.global_step)

    def apply_moving_average(self):
        variable_averages = tf.train.ExponentialMovingAverage(cfg.SOLVER.MOVING_AVERAGE_DECAY, self.global_step)
        self.variables_averages_op = variable_averages.apply(tf.trainable_variables())

    def get_train_op(self):
        with tf.control_dependencies([self.variables_averages_op, self.apply_gradient_op, self.bn_updates_op]):
            train_op = tf.no_op(name='train_op')
        return train_op
