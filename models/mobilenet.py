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

import tensorflow as tf
import tensorflow.contrib as tc


class MobileNetV2Rnn(object):
    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._normalizer = tc.layers.batch_norm

        self._lstm_layers = 2
        self._lstm_num_units = 256
        self._lstm_drop_ratio = 0.5
        self._ix = 0
        self._bn_params = None

        # self.logits, self.softmax = None, None
        # self.decoded, self.log_prob, self.seq_len = None, None, None

    def build(self, input_images, is_training=False):
        self._bn_params = {'is_training': is_training}
        with tf.variable_scope('crnn'):
            with tf.variable_scope('MobileNetV2'):
                x = tf.image.convert_image_dtype(input_images, tf.float32)
                x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)
                x = tc.layers.conv2d(x, 32, 3, 2, normalizer_fn=self._normalizer,
                                     normalizer_params=self._bn_params)  # 32 -> 16
                x = self._inverted_bottleneck(x, 1, 16, 0)
                x = self._inverted_bottleneck(x, 6, 24, 0)
                x = self._inverted_bottleneck(x, 6, 24, 0)
                x = self._inverted_bottleneck(x, 6, 32, 0, True)  # 16 -> 8
                x = self._inverted_bottleneck(x, 6, 32, 0)
                x = self._inverted_bottleneck(x, 6, 32, 0)
                # x = self._inverted_bottleneck(x, 6, 64, 1, False)  # 8 -> 4  4s
                x = self._inverted_bottleneck(x, 6, 64, 0, True)  # 8 -> 4  2s
                x = self._inverted_bottleneck(x, 6, 64, 0)
                x = self._inverted_bottleneck(x, 6, 64, 0)
                x = self._inverted_bottleneck(x, 6, 64, 0)
                x = self._inverted_bottleneck(x, 6, 96, 0, True)  # 4 -> 2
                x = self._inverted_bottleneck(x, 6, 96, 0)
                x = self._inverted_bottleneck(x, 6, 96, 0)
                x = self._inverted_bottleneck(x, 6, 160, 0, True)  # 2 -> 1
                x = self._inverted_bottleneck(x, 6, 160, 0)
                x = self._inverted_bottleneck(x, 6, 160, 0)
                x = self._inverted_bottleneck(x, 6, 320, 0)
                x = tc.layers.conv2d(x, 1024, 1, normalizer_fn=self._normalizer,
                                     activation_fn=tf.nn.relu6,
                                     normalizer_params=self._bn_params)
                x = tc.layers.conv2d(x, 1024, 1, activation_fn=None)  # (n, 1, w, num_class)
                x = tf.squeeze(x, axis=1)  # (n, w, c)
            logits = self._blstm(x, 0.5 if is_training else 1)  # [width(time), batch, n_classes]
        return logits

    def loss(self, logits, labels, seqlen):
        ctc_loss = tf.nn.ctc_loss(labels=labels,
                                  inputs=logits,
                                  sequence_length=seqlen,
                                  ignore_longer_outputs_than_inputs=True)
        ctc_loss_mean = tf.reduce_mean(ctc_loss)
        tf.add_to_collection('losses', ctc_loss_mean)
        losses = tf.get_collection('losses')
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
        return ctc_loss_mean, total_loss

    def eval(self, logits, labels, seqlen):
        softmax = tf.nn.softmax(logits, dim=-1, name='softmax')
        decoded, log_prob = tf.nn.ctc_greedy_decoder(softmax, seqlen)
        distance = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
        return distance

    def _inverted_bottleneck(self, inputs, up_sample_rate, channels, subsample, wsample=False):
        with tf.variable_scope('inverted_bottleneck{}_{}_{}'.format(self._ix, up_sample_rate, subsample)):
            self._ix += 1
            stride = 2 if subsample else 1
            wsample = [2, 1] if wsample else 1
            output = tc.layers.conv2d(inputs, up_sample_rate * inputs.get_shape().as_list()[-1], 1, stride=wsample,
                                      activation_fn=tf.nn.relu6,
                                      normalizer_fn=self._normalizer, normalizer_params=self._bn_params)

            output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride,
                                                activation_fn=tf.nn.relu6,
                                                normalizer_fn=self._normalizer, normalizer_params=self._bn_params)

            output = tc.layers.conv2d(output, channels, 1, activation_fn=None,
                                      normalizer_fn=self._normalizer, normalizer_params=self._bn_params)

            if inputs.get_shape().as_list()[-1] == channels:
                output = tf.add(inputs, output)
            return output

    def _blstm(self, inputs, keep_prob=1):
        with tf.variable_scope('LSTMLayers'):
            fw_cell_list = [tc.rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in
                            [self._lstm_num_units] * self._lstm_layers]
            bw_cell_list = [tc.rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in
                            [self._lstm_num_units] * self._lstm_layers]
            stack_lstm, _, _ = tc.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                      bw_cell_list,
                                                                      inputs,
                                                                      dtype=tf.float32)
            stack_lstm = tf.nn.dropout(stack_lstm, keep_prob=keep_prob)

            with tf.variable_scope('Reshaping_rnn'):
                shape = tf.shape(stack_lstm)
                rnn_reshaped = tf.reshape(stack_lstm, [shape[0] * shape[1], shape[2]])  # [batch x width, 2*n_hidden]

            with tf.variable_scope('fully_connected'):
                w = self._weight_var(shape=[self._lstm_layers * self._lstm_num_units, self._num_classes])
                b = self._bias_var(shape=[self._num_classes])
                fc_out = tf.nn.bias_add(tf.matmul(rnn_reshaped, w), b)

            lstm_out = tf.reshape(fc_out, [shape[0], shape[1], self._num_classes],
                                  name='reshape_out')  # [batch, width, n_classes]
            # Swap batch and time axis
            lstm_out = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]

            return lstm_out

    @staticmethod
    def _weight_var(shape, mean=0.0, stddev=0.02, name='weights'):
        # init = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
        return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev))

    @staticmethod
    def _bias_var(shape, value=0.0, name='bias'):
        # init = tf.constant(value=value, shape=shape)
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value=value))


if __name__ == '__main__':
    import os
    import numpy as np

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    MD = MobileNetV2RNN(75, is_training=True)
    fake_data = np.ones(shape=(2, 32, 150, 3))

    input_image = tf.placeholder(dtype=tf.uint8, shape=[2, 32, 150, 3], name='input_images')

    seq_len = np.ones(shape=[2], dtype=np.int32)
    seq = tf.convert_to_tensor(seq_len, tf.int32)

    paths, preds = MD.build_model(input_image)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _paths, _preds = sess.run([paths, preds], feed_dict={input_image: fake_data})
        print(_paths.shape)
        print(_preds.shape)
        print(_preds)
