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

import time
import tensorflow as tf

from core.config import cfg
from utils import logger


class Solver(object):
    def __init__(self):
        # Define the optimizer and its arguments
        self.optimizer = None
        self.opt_arguments = {
            'scale_gradient': 1. / (
                cfg.SOLVER.LOSS_SCALING *
                    cfg.SOLVER.ITER_SIZE),
            'clip_gradient': float(cfg.SOLVER.CLIP_NORM),
            'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
        }
        # Define the global step
        self.iter = 0
        # Define the decay step
        self._current_step = 0

    def _get_param_groups(self):
        param_groups = [
            {
                'params': [],
                'lr_mult': 1.,
                'decay_mult': 1.,
            },
            # Special treatment for biases (mainly to match historical impl.
            # details):
            # (1) Do not apply weight decay
            # (2) Use a 2x higher learning rate
            {
                'params': [],
                'lr_mult': 2.,
                'decay_mult': 0.,
            }
        ]
        for name, param in self.detector.named_parameters():
            if 'bias' in name: param_groups[1]['params'].append(param)
            else: param_groups[0]['params'].append(param)
        return param_groups

    def set_learning_rate(self):
        policy = cfg.SOLVER.LR_POLICY

        if policy == 'steps_with_decay':
            if self._current_step < len(cfg.SOLVER.STEPS) \
                    and self.iter >= cfg.SOLVER.STEPS[self._current_step]:
                self._current_step = self._current_step + 1
                logger.info('MultiStep Status: Iteration {}, step = {}' \
                    .format(self.iter, self._current_step))
                new_lr = cfg.SOLVER.BASE_LR * (
                        cfg.SOLVER.GAMMA ** self._current_step)
                self.optimizer.param_groups[0]['lr'] = \
                    self.optimizer.param_groups[1]['lr'] = new_lr
        else:
            raise ValueError('Unknown lr policy: ' + policy)

    def one_step(self):
        # Forward & Backward & Compute_loss
        iter_size = cfg.SOLVER.ITER_SIZE
        loss_scaling = cfg.SOLVER.LOSS_SCALING
        run_time = 0.; stats = {'loss': {'total': 0.}, 'iter': self.iter}
        add_loss = lambda x, y: y if x is None else x + y

        tic = time.time()

        if iter_size > 1:
            # Dragon is designed for manual gradients accumulating
            # ``zero_grad`` is only required if calling ``accumulate_grad``
            self.optimizer.zero_grad()

        for i in range(iter_size):
            outputs, total_loss = self.detector(), None
            # Sum the partial losses
            for k, v in outputs.items():
                if 'loss' in k:
                    if k not in stats['loss']:
                        stats['loss'][k] = 0.
                    total_loss = add_loss(total_loss, v)
                    stats['loss'][k] += float(v) * loss_scaling
            if loss_scaling != 1.: total_loss *= loss_scaling
            stats['loss']['total'] += float(total_loss)
            total_loss.backward()
            if iter_size > 1: self.optimizer.accumulate_grad()

        run_time += (time.time() - tic)

        # Apply Update
        self.set_learning_rate()
        tic = time.time()
        self.optimizer.step()
        run_time += (time.time() - tic)
        self.iter += 1

        # Average loss by the iter size
        for k in stats['loss'].keys():
            stats['loss'][k] /= cfg.SOLVER.ITER_SIZE

        # Misc stats
        stats['lr'] = self.base_lr
        stats['time'] = run_time
        return stats

    @property
    def base_lr(self):
        return self.optimizer.param_groups[0]['lr']

    @base_lr.setter
    def base_lr(self, value):
        self.optimizer.param_groups[0]['lr'] = \
            self.optimizer.param_groups[1]['lr'] = value


class SGDSolver(Solver):
    def __init__(self):
        super(SGDSolver, self).__init__()
        self.opt_arguments.update(**{
            'lr': cfg.SOLVER.BASE_LR,
            'momentum': cfg.SOLVER.MOMENTUM,
        })
        self.optimizer = torch.optim.SGD(
            self._get_param_groups(), **self.opt_arguments)


class NesterovSolver(Solver):
    def __init__(self):
        super(NesterovSolver, self).__init__()
        self.opt_arguments.update(**{
            'lr': cfg.SOLVER.BASE_LR,
            'momentum': cfg.SOLVER.MOMENTUM,
            'nesterov': True,
        })
        self.optimizer = torch.optim.SGD(
            self._get_param_groups(), **self.opt_arguments)


class RMSPropSolver(Solver):
    def __init__(self):
        super(RMSPropSolver, self).__init__()
        self.opt_arguments.update(**{
            'lr': cfg.SOLVER.BASE_LR,
            'alpha': 0.9,
            'eps': 1e-5,
        })
        self.optimizer = torch.optim.RMSprop(
            self._get_param_groups(), **self.opt_arguments)


class AdamSolver(Solver):
    def __init__(self):
        super(AdamSolver, self).__init__()
        self.opt_arguments.update(**{
            'lr': cfg.SOLVER.BASE_LR,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-5,
        })
        self.optimizer = torch.optim.RMSprop(
            self._get_param_groups(), **self.opt_arguments)


def get_solver_func(type):
    if type == 'MomentumSGD':
        return SGDSolver
    elif type == 'Nesterov':
        return NesterovSolver
    elif type == 'RMSProp':
        return RMSPropSolver
    elif type == 'Adam':
        return AdamSolver
    else:
        raise ValueError('Unsupported solver type: {}.\n'
            'Excepted in (MomentumSGD, Nesterov, RMSProp, Adam)'.format(type))
