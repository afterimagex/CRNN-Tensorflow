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
import shutil
import time
import numpy as np

from core.config import cfg, cfg_from_file


class Coordinator(object):
    """Coordinator is a simple tool to manage the
     unique experiments from the YAML configurations.

    """
    def __init__(self, cfg_file, exp_dir=None):
        # Override the default configs
        cfg_from_file(cfg_file)
        if cfg.EXP_DIR != '':
            exp_dir = cfg.EXP_DIR
        if exp_dir is None:
            model_id = time.strftime(
                '%Y%m%d_%H%M%S', time.localtime(time.time()))
            self.experiment_dir = '../experiments/{}'.format(model_id)
            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)
        else:
            if not os.path.exists(exp_dir):
                raise ValueError('ExperimentDir({}) does not exist.'.format(exp_dir))
            self.experiment_dir = exp_dir

    def _path_at(self, file, auto_create=True):
        path = os.path.abspath(os.path.join(self.experiment_dir, file))
        if auto_create and not os.path.exists(path): os.makedirs(path)
        return path

    def checkpoints_dir(self):
        return self._path_at('checkpoints')

    def summary_dir(self):
        return self._path_at('summary')

    def exports_dir(self):
        return self._path_at('exports')

    def results_dir(self, checkpoint=None):
        sub_dir = os.path.splitext(os.path.basename(checkpoint))[0] if checkpoint else ''
        return self._path_at(os.path.join('results', sub_dir))

    def checkpoint(self, global_step=None, wait=True):
        def locate():
            files = os.listdir(self.checkpoints_dir())
            steps = []
            for ix, file in enumerate(files):
                step = int(file.split('_iter_')[-1].split('.')[0])
                if global_step == step:
                    return os.path.join(self.checkpoints_dir(), files[ix]), step
                steps.append(step)
            if global_step is None:
                if len(files) == 0: return None, 0
                last_idx = int(np.argmax(steps)); last_step = steps[last_idx]
                return os.path.join(self.checkpoints_dir(), files[last_idx]), last_step
            return None, 0
        result = locate()
        while result[0] is None and wait:
            print('\rWaiting for step_{}.checkpoint to exist...'.format(global_step), end='')
            time.sleep(10)
            result = locate()
        return result

    def delete_experiment(self):
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)
