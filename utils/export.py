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
import tensorflow as tf
from models import get_models
from core.config import cfg
from utils import logger


def freeze_net(ckpt_path, save_path):
    from tensorflow.python.tools import freeze_graph

    logger.info('Freeze Model Will Saved at ', save_path)
    with tf.Graph().as_default():
        x1 = tf.placeholder(tf.float32, shape=[None, 32, None, 3], name='input_images')
        x2 = tf.placeholder(tf.int32, shape=[None], name='input_widths')

        logits = get_models(cfg.MODEL.BACKBONE)(cfg.MODEL.NUM_CLASSES).build(x1, False)
        seqlen = tf.cast(tf.floor_div(x2, 2), tf.int32, name='sequence_length')

        softmax = tf.nn.softmax(logits, dim=-1, name='softmax')
        decoded, log_prob = tf.nn.ctc_greedy_decoder(softmax, seqlen)

        prob_value = -1.0 * tf.reshape(log_prob, shape=[-1]) / tf.cast(seqlen, tf.float32)

        tf.identity(tf.cast(decoded[0].indices, dtype=tf.int32), name='indices')
        tf.identity(tf.cast(decoded[0].values, dtype=tf.int32), name='values')
        tf.identity(tf.shape(softmax)[0], name='length')
        tf.identity(prob_value, name='prob')

        saver = tf.train.Saver()
        with tf.Session(graph=tf.Graph()) as sess:
            saver.restore(sess, ckpt_path)
            fdir, name = os.path.split(save_path)
            tf.train.write_graph(sess.graph_def, fdir, name, as_text=True)

            freeze_graph.freeze_graph(
                input_graph=save_path,
                input_saver='',
                input_binary=False,
                input_checkpoint=ckpt_path,
                output_node_names='indices,values,prob,length',
                restore_op_name='',
                filename_tensor_name='',
                output_graph=save_path,
                clear_devices=True,
                initializer_nodes='',
            )
    logger.info('Freeze Model done.')


def save_model(ckpt_path, export_dir, version):
    export_path = os.path.join(tf.compat.as_bytes(export_dir), tf.compat.as_bytes(str(version)))

    logger.info('Exporting trained model to', export_path)

    with tf.Graph().as_default():
        x1 = tf.placeholder(tf.uint8, shape=[None, 32, None, 3], name='input_images')
        x2 = tf.placeholder(tf.int32, shape=[None], name='input_widths')  # for ctc_loss

        logits = get_models(cfg.MODEL.BACKBONE)(cfg.MODEL.NUM_CLASSES).build(x1, False)
        seqlen = tf.cast(tf.floor_div(x2, 2), tf.int32, name='sequence_length')

        softmax = tf.nn.softmax(logits, dim=-1, name='softmax')
        decoded, log_prob = tf.nn.ctc_greedy_decoder(softmax, seqlen)

        prob_value = -1.0 * tf.reshape(log_prob, shape=[-1]) / tf.cast(seqlen, tf.float32)

        y1 = tf.identity(decoded[0].indices, name='indices')
        y2 = tf.identity(decoded[0].values, name='values')
        y3 = tf.identity(prob_value, name='prob')

        saver = tf.train.Saver()
        with tf.Session(graph=tf.Graph()) as sess:
            saver.restore(sess, ckpt_path)

            legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")

            signature_def_map = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={
                            "input_images": tf.saved_model.utils.build_tensor_info(x1),
                            "input_widths": tf.saved_model.utils.build_tensor_info(x2),
                        },
                        outputs={
                            "indices": tf.saved_model.utils.build_tensor_info(y1),
                            "values": tf.saved_model.utils.build_tensor_info(y2),
                            "prob": tf.saved_model.utils.build_tensor_info(y3),
                        },
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                    )}

            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map=signature_def_map,
                legacy_init_op=legacy_init_op,
                assets_collection=None,
                clear_devices=True,
            )

            builder.save()
    logger.info('Export SaveModel done.')


def tflite(ckpt_path, save_path):
    logger.info('tflite Model Will Saved at ', save_path)
    with tf.Graph().as_default():
        x1 = tf.placeholder(tf.uint8, shape=[None, 32, 1024, 3], name='input_images')
        logits = get_models(cfg.MODEL.BACKBONE)(cfg.MODEL.NUM_CLASSES).build(x1, False)
        y1 = tf.nn.softmax(logits, dim=-1, name='softmax')

        saver = tf.train.Saver()
        with tf.Session(graph=tf.Graph()) as sess:
            saver.restore(sess, ckpt_path)
            converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [x1], [y1])
            tflite_model = converter.convert()
            with open(save_path, "wb") as wt:
                wt.write(tflite_model)

    logger.info('Convert to TFLite Model done.')
