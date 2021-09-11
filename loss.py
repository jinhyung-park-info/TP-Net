import tensorflow.keras.backend as K
import tensorflow as tf


def chamfer_distance_with_batch(p1, p2):

    p1 = tf.expand_dims(p1, axis=1)
    p2 = tf.expand_dims(p2, axis=1)

    p1 = tf.tile(p1, [1, K.shape(p2)[2], 1, 1])
    p1 = tf.transpose(p1, perm=[0, 2, 1, 3])

    p2 = tf.tile(p2, [1, K.shape(p1)[1], 1, 1])
    dist = tf.add(p1, tf.negative(p2))

    dist = tf.norm(dist, ord=2, axis=3)
    dist = tf.reduce_min(dist, axis=2)

    dist = tf.reduce_sum(dist) / tf.cast(K.shape(p1)[0], dtype='float32')

    return dist


def get_cd_loss(y_true, y_pred):
    dists_forward = chamfer_distance_with_batch(y_pred, y_true)
    dists_backward = chamfer_distance_with_batch(y_true, y_pred)
    return dists_forward + dists_backward


def get_weighted_cd_loss(y_true, y_pred):
    dists_forward = chamfer_distance_with_batch(y_pred, y_true)
    dists_backward = chamfer_distance_with_batch(y_true, y_pred)
    return (dists_forward + dists_backward) * 9.0


if __name__ == '__main__':

    pred_2d = [[1, 0], [2, 0], [100, 0]]
    gt_2d = [[1, 0], [2, 0], [101.5, 0]]

    gt_2d = tf.constant([gt_2d] * 512, dtype=tf.float32)
    pred_2d = tf.constant([pred_2d] * 512, dtype=tf.float32)

    print('Chamfer Loss: ')
    print(get_cd_loss(gt_2d, pred_2d))
