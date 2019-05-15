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


import cv2
import time
import PIL.Image
import PIL.ImageEnhance

import numpy as np
import tensorflow as tf
import numpy.random as npr

# tf.set_random_seed(int(time.time()))
np.random.seed(int(time.time()))


# class DataAugment(object):
#     def __init__(self):
#         pass
#
#     def augment(self, image):
#         with tf.name_scope('data_augment'):
#             image = self.tf_random_distort_color(image)
#             image = self.tf_random_channel_image(image)
#             image = self.cv2_random_rotate_image(image)
#             # image = self.tf_random_resize_image(image)
#             return image
#
#     @staticmethod
#     def tf_random_distort_color(image):
#         with tf.name_scope('tf_random_distort_color'):
#             image = tf.image.random_brightness(image, max_delta=0.3)
#             image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
#             image = tf.image.random_hue(image, max_delta=0.2)
#             image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
#             return image
#
#     @staticmethod
#     def tf_random_resize_image(image, min_scale=0.8, max_scale=1.2):
#         if image.get_shape().ndims != 3:
#             raise ValueError('\'image\' must have either 3 dimensions.')
#
#         with tf.name_scope('tf_random_resize_image'):
#             hscale = tf.random_uniform([], minval=min_scale, maxval=max_scale)
#             wscale = tf.random_uniform([], minval=min_scale, maxval=max_scale)
#             im_shape = tf.cast(tf.shape(image), tf.float32)
#             height = tf.cast(im_shape[0] * hscale, tf.int32)
#             width = tf.cast(im_shape[1] * wscale, tf.int32)
#             image = tf.image.resize_images(image, tf.stack([height, width]))
#             return image
#
#     @staticmethod
#     def tf_random_channel_image(image):
#         if image.get_shape().ndims != 3:
#             raise ValueError('\'image\' must have either 3 dimensions.')
#         with tf.name_scope('tf_random_channel_image'):
#             image = tf.transpose(image, [2, 0, 1])
#             image = tf.random_shuffle(image)
#             image = tf.transpose(image, [1, 2, 0])
#             return image
#
#     @staticmethod
#     def _cv2_rotate_image(im: np.ndarray, radian: np.float32):
#         if im.ndim != 3:
#             raise ValueError('image must have either 3 dimensions.')
#         h, w, c = im.shape
#         a = np.fabs(np.sin(radian))
#         b = np.fabs(np.cos(radian))
#         rw = int(h * a + w * b)
#         rh = int(w * a + h * b)
#         matrix = cv2.getRotationMatrix2D((w / 2., h / 2.), np.rad2deg(radian), 1.0)
#         matrix[0, 2] += (rw - w) / 2.
#         matrix[1, 2] += (rh - h) / 2.
#         image = cv2.warpAffine(im, matrix, (rw, rh), borderValue=0)
#         return image
#
#     def cv2_random_rotate_image(self, image):
#         if image.get_shape().ndims != 3:
#             raise ValueError('\'image\' must have either 3 dimensions.')
#         with tf.name_scope('cv2_random_rotate_image'):
#             radian = tf.random_uniform(shape=[], minval=-5, maxval=5) * np.pi / 180
#             image = tf.py_func(self._cv2_rotate_image, inp=[image, radian],
#                                Tout=[tf.uint8])[0]
#             image.set_shape([None, None, 3])
#             return image
#
#     @staticmethod
#     def tf_random_rotate_image(image):
#         if image.get_shape().ndims != 3:
#             raise ValueError('\'image\' must have either 3 dimensions.')
#         with tf.name_scope('tf_random_rotate_image'):
#             radians = tf.random_uniform(shape=[], minval=-5, maxval=5) * np.pi / 180
#             a = tf.abs(tf.sin(radians))
#             b = tf.abs(tf.cos(radians))
#             im_shape = tf.cast(tf.shape(image), tf.float32)
#             rw = tf.cast(im_shape[0] * a + im_shape[1] * b, tf.int32)
#             rh = tf.cast(im_shape[1] * a + im_shape[0] * b, tf.int32)
#             image = tf.image.resize_image_with_crop_or_pad(image,
#                                                            target_height=rh,
#                                                            target_width=rw)
#             image = tf.contrib.image.rotate(image, radians)
#             return image
#
#     @staticmethod
#     def tf_random_expand_image():
#         pass
#
#     @staticmethod
#     def _cv_random_salt_image(im: np.ndarray):
#         if im.ndim != 3:
#             raise ValueError('image must have either 3 dimensions.')
#         image = np.copy(im)
#         h, w, c = im.shape
#         num = np.random.randint(0, 50)
#         hs = np.random.randint(0, h, size=num)
#         ws = np.random.randint(0, w, size=num)
#         image[hs, ws, :] = np.random.randint(200, 255, 3)
#         return image
#
#     def cv_random_salt_image(self, image):
#         if image.get_shape().ndims != 3:
#             raise ValueError('\'image\' must have either 3 dimensions.')
#         with tf.name_scope('cv_random_salt_image'):
#             image = tf.py_func(self._cv_random_salt_image, inp=[image],
#                                Tout=[tf.uint8])[0]
#             image.set_shape([None, None, 3])
#             return image
#
#     @staticmethod
#     def _cv_random_line_image(im: np.ndarray):
#         image = np.copy(im)
#         h, w = im.shape[:2]
#         num = np.random.randint(1, 2)
#         if np.random.uniform() > 0.5:
#             begin = np.random.randint(0, w, size=num)
#             image[:, begin, :] = [255, 255, 255]
#         else:
#             begin = np.random.randint(0, h, size=num)
#             image[begin, :, :] = [255, 255, 255]
#         return image


class DataAugment(object):
    def __init__(self):
        pass

    def augment(self, im, prob=0.5):
        if npr.uniform() < prob:
            image = self.random_distort_image(im)
            process = [self.random_resize_image,
                       self.random_rotation_image, self.random_expand_image,
                       self.random_channel_image, self.random_salt, self.random_line]
            npr.shuffle(process)
            for p in process:
                if npr.uniform() < 0.5:
                    image = p(image)
            return image
        else:
            return im

    def random_distort_image(self, im):
        im = PIL.Image.fromarray(im)
        if npr.uniform() < 0.5:
            delta_brightness = npr.uniform(-0.3, 0.3) + 1.0
            im = PIL.ImageEnhance.Brightness(im)
            im = im.enhance(delta_brightness)
        if npr.uniform() < 0.5:
            delta_contrast = npr.uniform(-0.3, 0.3) + 1.0
            im = PIL.ImageEnhance.Contrast(im)
            im = im.enhance(delta_contrast)
        if npr.uniform() < 0.5:
            delta_saturation = npr.uniform(-0.3, 0.3) + 1.0
            im = PIL.ImageEnhance.Color(im)
            im = im.enhance(delta_saturation)
        im = np.array(im)
        return im

    @staticmethod
    def random_rotation_image(im, range=3):
        angle = npr.uniform(-range, range)
        h, w, c = im.shape
        a = np.sin(np.radians(angle))
        b = np.cos(np.radians(angle))
        rw = int(h * np.fabs(a) + w * np.fabs(b))
        rh = int(w * np.fabs(a) + h * np.fabs(b))
        # relative = ((rw - w) / 2, (rh - h) / 2)
        # ct = (rw / 2, rh / 2)
        # mean = np.mean(np.mean(im, axis=0), axis=0)
        Matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        Matrix[0, 2] += (rw - w) / 2
        Matrix[1, 2] += (rh - h) / 2

        image = cv2.warpAffine(im, Matrix, (rw, rh), borderValue=0)
        return image

    @staticmethod
    def random_blur_image(im):
        # h, w = im.shape[:2]
        # size_x = [i for i in range(int(w / 10)) if i % 2]
        # size_y = [i for i in range(int(h / 10)) if i % 2]
        size = [1, 3, 5]
        try:
            kernel_size = (npr.choice(size), npr.choice(size))
            sigma = npr.uniform(0, 3)
            image = cv2.GaussianBlur(im, kernel_size, sigma)
        except:
            image = im
        return image

    @staticmethod
    def random_resize_image(im, min_size=0.8, max_size=2):
        sigma_h = npr.uniform(min_size, max_size)
        sigma_w = npr.uniform(min_size, max_size)
        image = cv2.resize(im, (0, 0), fx=sigma_w, fy=sigma_h, interpolation=cv2.INTER_CUBIC)
        return image

    @staticmethod
    def random_expand_image(im):
        # mean = np.mean(np.mean(im, axis=0), axis=0)
        h, w = im.shape[:2]
        rh = int(h * npr.uniform(1, 1.25))
        rw = int(w * npr.uniform(1, 1.25))
        if rh == h:
            start_h = 0
        else:
            start_h = npr.randint(0, rh - h)
        if rw == w:
            start_w = 0
        else:
            start_w = npr.randint(0, rw - w)
        image = np.zeros((rh, rw, 3), dtype=np.uint8)  # * mean
        image[start_h:start_h + h, start_w:start_w + w, :] = im
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def random_channel_image(im):
        ch = [0, 1, 2]
        npr.shuffle(ch)
        image = im[:, :, ch]
        return image

    @staticmethod
    def random_salt(im):
        image = np.copy(im)
        h, w = im.shape[:2]
        num = npr.randint(0, 50)
        hs = npr.randint(0, h, size=num)
        ws = npr.randint(0, w, size=num)
        image[hs, ws, :] = npr.randint(200, 255, 3)
        return image

    @staticmethod
    def random_line(im):
        image = np.copy(im)
        h, w = im.shape[:2]
        num = npr.randint(1, 2)
        if npr.uniform() > 0.5:
            begin = npr.randint(0, w, size=num)
            image[:, begin, :] = [255, 255, 255]
        else:
            begin = npr.randint(0, h, size=num)
            image[begin, :, :] = [255, 255, 255]
        return image

    @staticmethod
    def randomAffine(im):
        image = np.copy(im)
        h, w = im.shape[:2]
        shift = npr.random_integers(0, w // 10, 4)
        origon = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        target = np.float32([[shift[0], 0], [w - shift[1], 0], [w - shift[2], h], [shift[3], h]])
        M = cv2.getPerspectiveTransform(origon, target)
        dst = cv2.warpPerspective(image, M, (w, h))
        return dst


def dataAugment(im):
    image = tf.py_func(DataAugment().augment, inp=[im],
                       Tout=[tf.uint8])[0]
    image.set_shape([None, None, 3])
    return image


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    image = cv2.imread(
        '/home/code/xupeichao/DATA/OCR_DATA_ALL_0427/DET_0506/image/ffe60c6b-9d24-46c0-87d4-4836a7123cd6_0.jpg')
    input_image = tf.placeholder(tf.uint8, shape=[None, None, 3])
    augment_image = data_augment(input_image)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        # while True:
        #
        #     # print(image_s)
        #     cv2.imshow('data augment', image_s[0])
        #     cv2.waitKey(0)
        ni = 3
        nj = 3
        for i in range(ni):
            for j in range(nj):
                image_s = sess.run([augment_image], feed_dict={input_image: image})[0]
                plt.subplot(ni, nj, i * ni + j + 1)
                plt.imshow(image_s[:, :, ::-1])
        plt.show()
