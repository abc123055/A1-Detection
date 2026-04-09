import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import cv2
import matplotlib
from src.flowlib import read_flow, flow_to_image
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

_preprocessing_ops = tf.load_op_library(
    tf.compat.v1.resource_loader.get_path_to_datafile("./src/ops/build/preprocessing.so"))


def display(img, c):
    plt.subplot(int('22' + str(c + 1)))
    plt.imshow(img[0, :, :, :])


def main():
    """
.Input("image_a: float32")
.Input("image_b: float32")
.Attr("crop: list(int) >= 2")
.Attr("params_a_name: list(string)")
.Attr("params_a_rand_type: list(string)")
.Attr("params_a_exp: list(bool)")
.Attr("params_a_mean: list(float32)")
.Attr("params_a_spread: list(float32)")
.Attr("params_a_prob: list(float32)")
.Attr("params_b_name: list(string)")
.Attr("params_b_rand_type: list(string)")
.Attr("params_b_exp: list(bool)")
.Attr("params_b_mean: list(float32)")
.Attr("params_b_spread: list(float32)")
.Attr("params_b_prob: list(float32)")
.Output("aug_image_a: float32")
.Output("aug_image_b: float32")
.Output("spatial_transform_a: float32")
.Output("inv_spatial_transform_b: float32")
    """

    crop = [364, 492]
    params_a_name = ['translate_x', 'translate_y']
    params_a_rand_type = ['uniform_bernoulli', 'uniform_bernoulli']
    params_a_exp = [False, False]
    params_a_mean = [0.0, 0.0]
    params_a_spread = [0.4, 0.4]
    params_a_prob = [1.0, 1.0]
    params_b_name = []
    params_b_rand_type = []
    params_b_exp = []
    params_b_mean = []
    params_b_spread = []
    params_b_prob = []

    with tf.compat.v1.Session() as sess:
        with tf.device('/gpu:0'):
            image_a = cv2.imread('./img0.ppm').astype(np.float64) / 255.0
            image_b = cv2.imread('./img1.ppm').astype(np.float64) / 255.0
            flow = read_flow('./flow.flo')

            image_a_tf = tf.expand_dims(tf.cast(tf.constant(image_a, dtype=tf.float64), tf.float32), 0)
            image_b_tf = tf.expand_dims(tf.cast(tf.constant(image_b, dtype=tf.float64), tf.float32), 0)

            preprocess = _preprocessing_ops.data_augmentation(image_a_tf,
                                                              image_b_tf,
                                                              crop,
                                                              params_a_name,
                                                              params_a_rand_type,
                                                              params_a_exp,
                                                              params_a_mean,
                                                              params_a_spread,
                                                              params_a_prob,
                                                              params_b_name,
                                                              params_b_rand_type,
                                                              params_b_exp,
                                                              params_b_mean,
                                                              params_b_spread,
                                                              params_b_prob)

            out = sess.run(preprocess)
            trans = out.spatial_transform_a
            inv_trans = out.inv_spatial_transform_b

            print(trans.shape)
            print(inv_trans.shape)

            flow_tf = tf.expand_dims(tf.cast(tf.constant(flow), tf.float32), 0)
            aug_flow_tf = _preprocessing_ops.flow_augmentation(flow_tf, trans, inv_trans, crop)

            aug_flow = sess.run(aug_flow_tf)[0, :, :, :]

            # Plot img0, img0aug
            plt.subplot(321)
            plt.imshow(image_a)
            plt.subplot(322)
            plt.imshow(out.aug_image_a[0, :, :, :])

            # Plot img1, img1aug
            plt.subplot(323)
            plt.imshow(image_b)
            plt.subplot(324)
            plt.imshow(out.aug_image_b[0, :, :, :])

            # Plot flow, flowaug
            plt.subplot(325)
            plt.imshow(flow_to_image(flow))
            plt.subplot(326)
            plt.imshow(flow_to_image(aug_flow))

            plt.show()


print(os.getpid())
input("Press Enter to continue...")
main()
