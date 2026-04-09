import tensorflow as tf


def intensity_loss(gen_frames, gt_frames, l_num):
    """
    Lp loss between generated frames and ground truth frames.

    :param gen_frames: tensor, generated frames.
    :param gt_frames: tensor, ground truth frames.
    :param l_num: int, 1 for L1 loss, 2 for L2 loss.
    :return: scalar tensor, mean intensity loss.
    """
    return tf.reduce_mean(tf.abs(gen_frames - gt_frames) ** l_num)


def gradient_loss(gen_frames, gt_frames, alpha):
    """
    Gradient Difference Loss (GDL).

    Penalizes differences in image gradients between generated and ground truth frames.

    :param gen_frames: tensor, generated frames [batch, h, w, c].
    :param gt_frames: tensor, ground truth frames [batch, h, w, c].
    :param alpha: int, exponent for gradient difference.
    :return: scalar tensor, mean gradient loss.
    """

    # gradients along x (horizontal) and y (vertical)
    gen_dx = tf.abs(gen_frames[:, 1:, :, :] - gen_frames[:, :-1, :, :])
    gen_dy = tf.abs(gen_frames[:, :, 1:, :] - gen_frames[:, :, :-1, :])
    gt_dx = tf.abs(gt_frames[:, 1:, :, :] - gt_frames[:, :-1, :, :])
    gt_dy = tf.abs(gt_frames[:, :, 1:, :] - gt_frames[:, :, :-1, :])

    grad_diff_x = tf.abs(gt_dx - gen_dx) ** alpha
    grad_diff_y = tf.abs(gt_dy - gen_dy) ** alpha

    return tf.reduce_mean(grad_diff_x) + tf.reduce_mean(grad_diff_y)
