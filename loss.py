import tensorflow.keras.backend as K
from common.Constants import NUM_PARTICLES
import tensorflow as tf


def chamfer_distance_and_shape_with_batch(p1, p2, debug=False):
    p1 = tf.expand_dims(p1, axis=1)
    p2 = tf.expand_dims(p2, axis=1)

    p1 = tf.tile(p1, [1, K.shape(p2)[2], 1, 1])
    transposed_p1 = tf.transpose(p1, perm=[0, 2, 1, 3])

    # Chamfer Distance
    p2 = tf.tile(p2, [1, K.shape(transposed_p1)[1], 1, 1])
    chamfer_dist = tf.add(transposed_p1, tf.negative(p2))

    chamfer_dist = tf.norm(chamfer_dist, ord=2, axis=3)
    chamfer_dist = tf.reduce_min(chamfer_dist, axis=2)

    chamfer_dist = tf.reduce_sum(chamfer_dist) / tf.cast(K.shape(transposed_p1)[0], dtype='float32')

    # Shape Loss
    particle_dist = tf.add(transposed_p1, tf.negative(p1))
    particle_dist = -tf.norm(particle_dist, ord='euclidean', axis=3)

    particle_dist = -tf.math.top_k(particle_dist, k=3).values[:, :, 1:]
    particle_dist = tf.reshape(particle_dist, shape=(-1, NUM_PARTICLES * 2))

    variances = tf.math.reduce_variance(particle_dist, axis=1)
    mean_variance = tf.reduce_mean(variances)

    return chamfer_dist + 20000 * mean_variance


def chamfer_distance_with_batch(p1, p2, debug=False):
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


def squared_chamfer_distance_with_batch(p1, p2, debug=False):
    p1 = tf.expand_dims(p1, axis=1)
    p2 = tf.expand_dims(p2, axis=1)

    p1 = tf.tile(p1, [1, K.shape(p2)[2], 1, 1])
    p1 = tf.transpose(p1, perm=[0, 2, 1, 3])

    p2 = tf.tile(p2, [1, K.shape(p1)[1], 1, 1])
    dist = tf.add(p1, tf.negative(p2))

    dist = tf.norm(dist, ord=2, axis=3)
    dist = tf.square(dist)
    dist = tf.reduce_min(dist, axis=2)

    dist = tf.reduce_sum(dist) / tf.cast(K.shape(p1)[0], dtype='float32')

    return dist


def get_cd_loss_func(y_true, y_pred):
    copied_y_pred = y_pred
    copied_y_true = y_true
    dists_forward = chamfer_distance_with_batch(y_pred, y_true)
    dists_backward = chamfer_distance_with_batch(copied_y_true, copied_y_pred)
    return dists_forward + dists_backward


def get_cd_loss_func_for_first(y_true, y_pred):
    copied_y_pred = y_pred
    copied_y_true = y_true
    dists_forward = chamfer_distance_with_batch(y_pred, y_true)
    dists_backward = chamfer_distance_with_batch(copied_y_true, copied_y_pred)
    return (dists_forward + dists_backward) * 9.0


def get_squared_cd_loss_func(y_true, y_pred):
    copied_y_pred = y_pred
    copied_y_true = y_true
    dists_forward = squared_chamfer_distance_with_batch(y_pred, y_true)
    dists_backward = squared_chamfer_distance_with_batch(copied_y_true, copied_y_pred)
    return dists_forward + dists_backward


def get_squared_cd_loss_func_for_first(y_true, y_pred):
    copied_y_pred = y_pred
    copied_y_true = y_true
    dists_forward = squared_chamfer_distance_with_batch(y_pred, y_true)
    dists_backward = squared_chamfer_distance_with_batch(copied_y_true, copied_y_pred)
    return (dists_forward + dists_backward) * 9.0

def mean_absolute_error(y_true, y_pred):
    loss = K.mean(K.abs(y_true - y_pred))
    return loss


def mae_for_first(y_true, y_pred):
    loss = K.mean(K.abs(y_true - y_pred))
    return loss * 9.0

def mae_base(y_true, y_pred):
    loss = K.mean(K.abs(y_true - y_pred))
    return loss


def chamfer_and_mae(y_true, y_pred):
    return get_cd_loss_func(y_true, y_pred) * 0.05 + mean_absolute_error(y_true, y_pred)


def chamfer_and_mae_for_first(y_true, y_pred):
    return chamfer_and_mae(y_true, y_pred) * 9.0

def chamfer_and_mse_for_first(y_true, y_pred):
    return chamfer_and_mse(y_true, y_pred) * 9.0

def chamfer_and_mse(y_true, y_pred):
    return get_cd_loss_func(y_true, y_pred) * 0.02 + tf.keras.losses.mean_squared_error(y_true, y_pred)


def chamfer_and_shape(y_true, y_pred):
    copied_y_pred = y_pred
    copied_y_true = y_true
    dists_forward_and_shape_loss = chamfer_distance_and_shape_with_batch(y_pred, y_true)
    dists_backward = chamfer_distance_with_batch(copied_y_true, copied_y_pred)
    return dists_forward_and_shape_loss + dists_backward


def chamfer_and_shape_for_first(y_true, y_pred):
    return chamfer_and_shape(y_true, y_pred) * 9.0


def mse_for_first(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred) * 9.0


def mse_base(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


def real_chamfer_distance(y_true, y_pred):
    return chamfer_distance_with_batch(y_true, y_pred)


def real_chamfer_distance_for_first(y_true, y_pred):
    return chamfer_distance_with_batch(y_true, y_pred) * 9.0


if __name__ == '__main__':
    """
    pred_2d = [[0.2398509979248047, 0.07136905193328857], [0.08903583884239197, 0.10960081219673157], [0.06465977430343628, 0.1877201795578003],
               [0.31310927867889404, 0.17703306674957275], [0.1716098189353943, 0.06653550267219543], [0.11360758543014526, 0.24200409650802612],
               [0.0016888976097106934, 0.09439757466316223], [0.04867035150527954, 0.22318214178085327], [0.11003592610359192, 0.08354192972183228],
               [0.09700265526771545, 0.14013069868087769], [0.29239365458488464, 0.11442121863365173], [0.2916931211948395, 0.23942404985427856],
               [0.11415272951126099, 0.314358651638031], [0.18305933475494385, 0.014288336038589478], [0.11200481653213501, 0.2868407666683197],
               [0.26854273676872253, 0.26408156752586365], [0.30643099546432495, 0.2085258662700653], [0.10177072882652283, 0.00425565242767334],
               [0.17026588320732117, 0.28193360567092896], [0.19096800684928894, 0.17278262972831726]]

    gt_2d = [[0.31385555555555555, 0.17790555555555557], [0.30938333333333334, 0.14518333333333333], [0.30852777777777773, 0.2103722222222222],
             [0.29533333333333334, 0.11499444444444445], [0.2939, 0.23987222222222224], [0.27112222222222226, 0.26388333333333336],
             [0.2726388888888889, 0.09006666666666667], [0.24303333333333332, 0.07289999999999999], [0.2419666666666667, 0.2802277777777778],
             [0.20882222222222221, 0.28718333333333335], [0.20899999999999996, 0.0655222222222222], [0.19256666666666666, 0.17360555555555557],
             [0.17458333333333334, 0.2836944444444444], [0.17374444444444442, 0.0691888888888889], [0.14252777777777778, 0.2695666666666667],
             [0.14094444444444443, 0.08410555555555556], [0.11435555555555554, 0.10911666666666668], [0.11606666666666667, 0.24567777777777777],
             [0.09833888888888888, 0.2141111111111111], [0.09170555555555555, 0.17808888888888885]]
    """


    gt_2d = [[1, 0], [2, 0]]
    pred_2d = [[1, 0], [1, 0]]

    gt_2d = tf.constant([gt_2d] * 512, dtype=tf.float32)
    pred_2d = tf.constant([pred_2d] * 512, dtype=tf.float32)

    import time

    tic = time.time()
    print('Chamfer Loss: ')
    print(get_cd_loss_func(gt_2d, pred_2d))
    print('Real Chamfer Loss: ')
    print(real_chamfer_distance(gt_2d, pred_2d))
    #toc = time.time() - tic
    #print("elapsed=", toc)  #




