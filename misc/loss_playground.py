import tensorflow as tf
import tensorflow.keras.backend as K
from misc.tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from misc.tf_ops.grouping import query_ball_point, group_point, query_ball_point_var_rad
from misc.tf_ops.emd import tf_auctionmatch
import numpy as np


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


def get_cd_loss_func(y_pred, y_true):
    copied_y_pred = y_pred
    copied_y_true = y_true
    dists_forward = chamfer_distance_with_batch(y_pred, y_true)
    dists_backward = chamfer_distance_with_batch(copied_y_true, copied_y_pred)
    return dists_forward + dists_backward


def get_cd_loss_func_for_first(y_pred, y_true):
    copied_y_pred = y_pred
    copied_y_true = y_true
    dists_forward = chamfer_distance_with_batch(y_pred, y_true)
    dists_backward = chamfer_distance_with_batch(copied_y_true, copied_y_pred)
    return (dists_forward + dists_backward) * 7.0


def mean_absolute_error(y_pred, y_true):
    loss = K.mean(K.abs(y_true - y_pred))
    return loss


def chamfer_and_mae(y_pred, y_true):
    return get_cd_loss_func(y_pred, y_true) * 0.05 + mean_absolute_error(y_pred, y_true)


def get_repulsion_loss4(pred, nsample=7, radius=0.2):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    # tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ## get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss


def get_emd_loss3D(pred, gt):
    batch_size = K.shape(pred)[0]
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = gather_point(gt, matchl_out)
    dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    dist = tf.reduce_mean(input_tensor=dist, axis=1, keepdims=True)
    emd_loss = tf.reduce_mean(input_tensor=dist)
    return emd_loss  # emd_loss


def get_emd_rep_loss(y_pred, y_true):
    emd_loss = get_emd_loss3D(y_pred, y_true)
    uniform_loss = get_repulsion_loss4(y_pred)
    return emd_loss + uniform_loss

"""
def compute_distance_variance(pointset):
    extended_pointset = tf.reshape(tf.repeat(pointset, repeats=[NUM_PARTICLES] * NUM_PARTICLES, axis=0), shape=(NUM_PARTICLES, NUM_PARTICLES, 2))
    distances = -tf.math.top_k(-tf.reduce_sum(tf.square(tf.add(extended_pointset, -pointset)), axis=-1), k=3).values[:, 1:]
    distances = tf.reshape(distances, shape=[NUM_PARTICLES * 2])
    return tf.math.reduce_variance(distances)


def shape_loss(y_pred):
    variances = tf.vectorized_map(fn=compute_distance_variance, elems=y_pred)
    y_pred_variances = tf.reduce_mean(variances)
    return y_pred_variances


def chamfer_and_shape(y_true, y_pred):
    return get_cd_loss_func(y_pred, y_true) + 200 * shape_loss(y_pred)
    
def shape_loss_with_batch(p2):
    p2 = tf.expand_dims(p2, axis=1)
    p2 = tf.tile(p2, [1, NUM_PARTICLES, 1, 1])
    transposed_p2 = tf.transpose(p2, perm=[0, 2, 1, 3])

    dist = tf.add(transposed_p2, tf.negative(p2))
    dist = -tf.norm(dist, ord='euclidean', axis=3)

    dist = -tf.math.top_k(dist, k=3).values[:, :, 1:]
    dist = tf.reshape(dist, shape=(-1, NUM_PARTICLES * 2))

    variances = tf.math.reduce_variance(dist, axis=1)

    return tf.reduce_mean(variances)    
"""

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    p1 = tf.compat.v1.placeholder(tf.float32, shape=(1, 21, 2))
    p2 = tf.compat.v1.placeholder(tf.float32, shape=(1, 21, 2))
    p3 = tf.compat.v1.placeholder(tf.float32, shape=(1, 21, 3))
    p4 = tf.compat.v1.placeholder(tf.float32, shape=(1, 21, 3))
    p5 = tf.compat.v1.placeholder(tf.float32, shape=(1, 20, 3))

    cd_2D_gpu = get_cd_loss_func(p1, p2)
    emd_3D_gpu = get_emd_loss3D(p3, p4)
    rp_3D_gpu = get_repulsion_loss4(p3)
    rp_sat_3D_gpu = get_repulsion_loss4(p5)
    emd_rep_3D_gpu = get_emd_rep_loss(p3, p4)

    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05))
    sess = tf.compat.v1.Session(config=config)

    pred_2d = [[0.2398509979248047, 0.07136905193328857], [0.08903583884239197, 0.10960081219673157], [0.06465977430343628, 0.1877201795578003],
               [0.31310927867889404, 0.17703306674957275], [0.1716098189353943, 0.06653550267219543], [0.11360758543014526, 0.24200409650802612],
               [0.0016888976097106934, 0.09439757466316223], [0.04867035150527954, 0.22318214178085327], [0.11003592610359192, 0.08354192972183228],
               [0.09700265526771545, 0.14013069868087769], [0.29239365458488464, 0.11442121863365173], [0.2916931211948395, 0.23942404985427856],
               [0.11415272951126099, 0.314358651638031], [0.18305933475494385, 0.014288336038589478], [0.11200481653213501, 0.2868407666683197],
               [0.26854273676872253, 0.26408156752586365], [0.30643099546432495, 0.2085258662700653], [0.10177072882652283, 0.00425565242767334],
               [0.17026588320732117, 0.28193360567092896], [0.19096800684928894, 0.17278262972831726], [0.015457212924957275, 0.002765297656878829]]

    gt_2d = [[0.31385555555555555, 0.17790555555555557], [0.30938333333333334, 0.14518333333333333], [0.30852777777777773, 0.2103722222222222],
             [0.29533333333333334, 0.11499444444444445], [0.2939, 0.23987222222222224], [0.27112222222222226, 0.26388333333333336],
             [0.2726388888888889, 0.09006666666666667], [0.24303333333333332, 0.07289999999999999], [0.2419666666666667, 0.2802277777777778],
             [0.20882222222222221, 0.28718333333333335], [0.20899999999999996, 0.0655222222222222], [0.19256666666666666, 0.17360555555555557],
             [0.17458333333333334, 0.2836944444444444], [0.17374444444444442, 0.0691888888888889], [0.14252777777777778, 0.2695666666666667],
             [0.14094444444444443, 0.08410555555555556], [0.11435555555555554, 0.10911666666666668], [0.11606666666666667, 0.24567777777777777],
             [0.09833888888888888, 0.2141111111111111], [0.09170555555555555, 0.17808888888888885], [0.09722777777777779, 0.14167777777777776]]

    pred_3d = []
    gt_3d = []

    for coordinate in pred_2d:
        pred_3d.append(coordinate + [0])
    for coordinate in gt_2d:
        gt_3d.append(coordinate + [0])

    gt_2d = np.array([gt_2d])
    pred_2d = np.array([pred_2d])
    gt_3d = np.array([gt_3d])
    pred_3d = np.array([pred_3d])

    print('Chamfer Distance Loss: ')
    print(sess.run(cd_2D_gpu, feed_dict={p1: gt_2d, p2: pred_2d}))

    print('EMD Distance Loss: ')
    print(sess.run(emd_3D_gpu, feed_dict={p3: gt_3d, p4: pred_3d}))

    print('Repulsion Loss of Prediction: ')
    print(sess.run(rp_3D_gpu, feed_dict={p3: pred_3d}))

    print('Repulsion Loss of Ground Truth: ')
    print(sess.run(rp_3D_gpu, feed_dict={p3: gt_3d}))

    print('EMD + Rep Loss: ')
    print(sess.run(emd_rep_3D_gpu, feed_dict={p3: pred_3d, p4: gt_3d}))
