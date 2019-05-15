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

import os
import csv
import numpy as np
import tensorflow as tf


class Character(list):
    def __init__(self, *args):
        list.__init__(self, *args)
        tmp = sorted(list(set(self)))
        self.clear()
        self.extend(tmp)

    def __iadd__(self, *args, **kwargs):
        value = self.__add__(*args, **kwargs)
        self.update(value)
        return self

    def update(self, value):
        self.clear()
        tmp = sorted(list(set(value)))
        self.extend(tmp)
        return self

    def encode(self, char):
        return self.index(char)

    def decode(self, idx):
        return self[idx]

    def sparse_to_strlist(self, indices, values, batchsize):
        '''
        :param indices: tf.SparseTensor.indices
        :param values: tf.SparseTensor.values
        :param batchsize:
        :return:
        '''
        decoded_indexes = {}

        for i, idx in enumerate(indices):
            if idx[0] not in decoded_indexes:
                decoded_indexes[idx[0]] = []
            decoded_indexes[idx[0]].append(values[i])

        mi = len(self)
        for k, v in decoded_indexes.items():
            decoded_indexes[k] = ''.join([self[i] if i < mi else ' {} '.format(str(i)) for i in v])

        if len(decoded_indexes) < batchsize:
            for i in range(len(decoded_indexes), batchsize):
                decoded_indexes[i] = ''

        if len(decoded_indexes) > 0:
            max_idx = max(decoded_indexes.keys()) + 1
            words = ['' for i in range(max_idx)]
            for k, v in decoded_indexes.items():
                words[k] = v
            return words
        else:
            return []

    def from_txt(self, text_file):
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                value = f.read().splitlines()
                self.update(value)
        return self

    def write_to_txt(self, text_file: str):
        with open(text_file, 'w', encoding='utf-8') as f:
            f.writelines([ac + os.linesep for ac in self])

    @staticmethod
    def list_to_sparse(sequences, dtype=np.int32):
        """
        Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape


if __name__ == "__main__":
    csv_file = '/home/code/xupeichao/DATA/crnn_ktp_npwp_sim_tfrecords/character.csv'
    e = Character().from_csv(csv_file)
    print(len(e))
    print(e)

    print([e.index(i) for i in 'ABC'])
    print([e.index(i) for i in '123'])
    print([e.index(i) for i in 'Test'])

    print(e.encode('A'))

    s = [[25, 26, 27], [12, 13, 14], [44, 57, 69, 70]]
    p = tf.SparseTensorValue(*sparse_tuple_from(s))
    d = e.decode(p.indices, p.values, len(s))
    print(d)
