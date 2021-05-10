import tensorflow as tf
from tensorflow.keras.layers import ReLU, Conv2D, BatchNormalization
import tensorflow.keras.backend as K
import numpy as np
from misc.tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from misc.tf_ops.grouping import query_ball_point, group_point, query_ball_point_var_rad, knn_point
from misc.tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XY) in local regions
    '''
    print('Before sample and group gather point')
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    print('Passed sample and group gather point')
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = K.shape(xyz)[0]
    # batch_size = xyz.get_shape()[0]
    nsample = xyz.get_shape()[1]
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size, 1, 1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, group_all, bn=False, pooling='max', knn=False, is_training=True, use_xyz=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    # Sample and Grouping
    if group_all:
        nsample = xyz.get_shape()[1]
        new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
    else:
        new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

    print('new points shape before mlp: {}'.format(new_points.shape))
    # Point Feature Embedding
    for i, num_out_channel in enumerate(mlp):
        new_points = Conv2D(filters=num_out_channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(new_points)   # conv2d(inputs, kernel) + bias with bias_initializer='zeros'
        if bn:
            if is_training:
                new_points = BatchNormalization(center=True, scale=True, momentum=0.9)(training=True, inputs=new_points)
            else:
                new_points = BatchNormalization(center=True, scale=True, momentum=0.9)(inputs=new_points)
        new_points = ReLU()(new_points)
        print('new_points shape after {}th Conv2D : {}'.format(i+1, new_points.shape))
    # if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

    # Pooling in Local Regions
    if pooling=='max':
        #if not group_all:
        new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
        #else:
        #    new_points = tf.reduce_max(new_points, axis=[1], keepdims=True, name='maxpool')
    else:
        exit(0)
    #if not group_all:
    new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
    #else:
    #    new_points = tf.squeeze(new_points, [1])
    print('Return new points shape : {}'.format(new_points.shape))
    print('Return new_xyz shape : {}'.format(new_xyz.shape))
    return new_xyz, new_points, idx


def meteor_direct_module(xyz, time, points, npoint, radius, nsample, mlp, group_all, is_training=True, module_type='ind', fps=True, bn=False, knn=False, use_xyz=True):
    '''
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            time: (batch_size, 63, 1) TF tensor - t-2, t-1, t
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radiuses in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            module_type: 'ind' or 'rel' -- the type of meteor module
            fps: whether to do farthest point sampling; Requires npoint == xyz.get_shape()[1].value, when fps=False
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    sample_idx = None
    batch_size = K.shape(xyz)[0]
    #batch_size = xyz.get_shape()[0]

    if fps:
        ##### sample and group with variable radius
        sample_idx = farthest_point_sample(npoint, xyz)
    else:
        ##### no sampling at all
        sample_idx = tf.tile(tf.expand_dims(tf.range(npoint, dtype=tf.int32), 0), [batch_size, 1])

    new_xyz = gather_point(xyz, sample_idx) # (batch_size, npoint, 3)
    print('Passed first gather point')
    new_time = gather_point(time, sample_idx) # (batch_size, npoint, 1)
    print('Passed second gather point')
    new_points = gather_point(points, sample_idx) # (batch_size, npoint, channel)
    print('Passed third gather point')
    time_ = tf.reshape(time, [batch_size, 1, -1]) # (batch_size, 1, ndataset)
    new_time_ = tf.abs(new_time - time_) # (batch_size, npoint, ndataset)

    radius_ = tf.gather(radius, tf.cast(new_time_, tf.int32)) # (batch_size, npoint, ndataset)

    # TODO : query ball point var rad == query ball point
    idx, pts_cnt = query_ball_point_var_rad(radius_, nsample, xyz, new_xyz)
    # idx, pts_cnt = query_ball_point(radius_, nsample, xyz, new_xyz)

    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        grouped_time = group_point(time, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            if module_type == 'ind':
                print('Ind Module before new_points: {}'.format(new_points.shape))
                new_points = tf.concat([grouped_xyz, grouped_time, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+1+channel)
                print('Ind Module after new_points: {}'.format(new_points.shape))
            else:
                print('Rel Module before new_points: {}'.format(new_points.shape))
                new_points_expand = tf.tile(tf.expand_dims(new_points, 2), [1,1,nsample,1])
                print('Rel Module before new_points_expand: {}'.format(new_points_expand.shape))
                new_points = tf.concat([grouped_xyz, grouped_time, grouped_points, new_points_expand], axis=-1) # (batch_size, npoint, nample, 3+1+channel+channel)
                print('Rel Module after new_points: {}'.format(new_points.shape))
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    # Point Feature Embedding
    for i, num_out_channel in enumerate(mlp):
        new_points = Conv2D(filters=num_out_channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(new_points)
        # conv2d(inputs, kernel) + bias with bias_initializer='zeros'
        if bn:
            if is_training:
                new_points = BatchNormalization(center=True, scale=True, momentum=0.9)(training=True, inputs=new_points)
            else:
                new_points = BatchNormalization(center=True, scale=True, momentum=0.9)(inputs=new_points)
        new_points = ReLU()(new_points)
        print('new_points shape after {}th Conv2D : {}'.format(i+1, new_points.shape))

    new_points = tf.reduce_max(new_points, axis=[2], name='maxpool')
    return new_xyz, new_time, new_points, idx


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training=True, bn=False, last=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    dist, idx = three_nn(xyz1, xyz2)
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0/dist),axis=2, keepdims=True)
    norm = tf.tile(norm,[1,1,3])
    weight = (1.0/dist) / norm
    interpolated_points = three_interpolate(points2, idx, weight)

    if points1 is not None:
        new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
    else:
        new_points1 = interpolated_points
    new_points1 = tf.expand_dims(new_points1, 2)

    for i, num_out_channel in enumerate(mlp):
        if last and i == len(mlp) - 1:
            new_points1 = Conv2D(filters=num_out_channel, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=tf.nn.sigmoid)(new_points1)
        else:
            new_points1 = Conv2D(filters=num_out_channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(new_points1)
            # conv2d(inputs, kernel) + bias with bias_initializer='zeros'
            if bn:
                if is_training:
                    new_points1 = BatchNormalization(center=True, scale=True, momentum=0.9)(training=True, inputs=new_points1)
                else:
                    new_points1 = BatchNormalization(center=True, scale=True, momentum=0.9)(inputs=new_points1)
            new_points1 = ReLU()(new_points1)
        print('new_points shape after {}th Conv2D : {}'.format(i+1, new_points1.shape))

    new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
    return new_points1
