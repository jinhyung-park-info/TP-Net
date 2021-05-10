from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Dropout, UpSampling2D, MaxPooling2D
from common.Constants import *
from tf_utils import *


def build_encoder():

    grey_image = Input(shape=(256, 256))
    net = tf.expand_dims(grey_image, axis=-1)

    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = MaxPooling2D(pool_size=(2, 2),
                       data_format='channels_last')(net)

    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = MaxPooling2D(pool_size=(2, 2),
                       data_format='channels_last')(net)

    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = MaxPooling2D(pool_size=(2, 2),
                       data_format='channels_last')(net)

    net = Flatten()(net)
    net = Dropout(0.4)(net)
    net = Dense(NUM_PARTICLES * 2, activation=tf.nn.sigmoid)(net)
    pointset = Reshape(target_shape=(NUM_PARTICLES, 2))(net)
    encoder = Model(inputs=grey_image, outputs=pointset)
    return encoder


def build_decoder():

    pointset = Input(shape=(NUM_PARTICLES, 2))

    net = Flatten()(pointset)
    net = Dense(units=1024, activation=tf.nn.relu)(net)
    net = Reshape(target_shape=(32, 32, 1))(net)

    net = Conv2D(filters=16,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=16,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=16,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = UpSampling2D(size=(2, 2),
                       data_format='channels_last')(net)

    net = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = UpSampling2D(size=(2, 2),
                       data_format='channels_last')(net)

    net = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = UpSampling2D(size=(2, 2), data_format='channels_last')(net)

    net = Conv2D(filters=1,
                 kernel_size=(3, 3),
                 activation=tf.nn.sigmoid,
                 padding='same')(net)

    reconstructed_grey_image = Reshape(target_shape=(256, 256))(net)

    decoder = Model(inputs=pointset, outputs=reconstructed_grey_image)
    return decoder


def build_decoder_for_high_res_image():
    pointset = Input(shape=(NUM_PARTICLES, 2))

    net = Flatten()(pointset)
    net = Dense(units=900, activation=tf.nn.leaky_relu)(net)
    net = Reshape(target_shape=(30, 30, 1))(net)

    net = Conv2D(filters=16,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=16,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = UpSampling2D(size=(4, 4),                    # 120, 120
                       data_format='channels_last')(net)

    net = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = UpSampling2D(size=(2, 2),
                       data_format='channels_last')(net)

    net = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = UpSampling2D(size=(2, 2), data_format='channels_last')(net)

    net = Conv2D(filters=1,
                 kernel_size=(3, 3),
                 activation=tf.nn.tanh,
                 padding='same')(net)

    reconstructed_grey_image = Reshape(target_shape=(480, 480))(net)

    decoder = Model(inputs=pointset, outputs=reconstructed_grey_image)
    return decoder


def build_meteornet(batch_size):

    # ============================= PointNet =============================
    point_cloud_input = Input(shape=(NUM_PARTICLES, 3), name='point_cloud_input')

    net = tf.expand_dims(point_cloud_input, axis=-1)
    net = Conv2D(filters=64,
                 kernel_size=(1, 3),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    local_features = Conv2D(filters=64,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation=tf.nn.relu,
                            padding='valid',
                            data_format='channels_last')(net)

    net = Conv2D(filters=64,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(local_features)

    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    net = Conv2D(filters=512,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    net = MaxPooling2D(size=(NUM_PARTICLES, 1),
                       strides=(2, 2),
                       padding='valid')(net)

    global_features = Reshape(target_shape=(512, ), name='global_features')(net)

    global_features_extended = tf.tile(global_features, [1, NUM_PARTICLES])
    global_features_extended = Reshape(target_shape=(NUM_PARTICLES, 1, 512), name='global_features_extended')(global_features_extended)

    total_features = tf.concat([local_features, global_features_extended], axis=-1)

    net = Conv2D(filters=256,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(total_features)

    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    point_features = Reshape(target_shape=(NUM_PARTICLES, 128), name='point_features')(net)

    feature_extractor = Model(inputs=point_cloud_input, outputs=[point_cloud_input, point_features])

    # ============================= Uses PointNet to prepare input for MeteorNet =============================
    # Shares the PointNet Model (feature extractor) for each timestep

    meteor_net_input = Input(shape=(NUM_INPUT_FRAMES, NUM_PARTICLES, 3), batch_size=batch_size, name='meteornet_input')

    l0_xyz, feature_set_a = feature_extractor(meteor_net_input[:, 0])
    l1_xyz, feature_set_b = feature_extractor(meteor_net_input[:, 1])
    l2_xyz, feature_set_c = feature_extractor(meteor_net_input[:, 2])

    l0_xyz = tf.concat([l0_xyz, l1_xyz, l2_xyz], axis=-2)
    l0_points = tf.concat([feature_set_a, feature_set_b, feature_set_c], axis=-2)
    l0_time = tf.concat([tf.ones([batch_size, NUM_PARTICLES, 1]) * i for i in range(NUM_INPUT_FRAMES)], axis=-2)
    l0_points = tf.concat([l0_points, l0_time], axis=-1)

    RADIUS = np.array([0.2, 0.25, 0.3], dtype='float32')

    # ================================ MeteorNet ===============================
    l1_xyz, l1_time, l1_points, l1_indices = meteor_direct_module(l0_xyz,
                                                                  l0_time,
                                                                  l0_points,
                                                                  npoint=NUM_PARTICLES * NUM_INPUT_FRAMES,
                                                                  module_type='ind',
                                                                  radius=RADIUS,
                                                                  nsample=7 * NUM_INPUT_FRAMES,
                                                                  mlp=[64, 64, 128],
                                                                  group_all=True,
                                                                  knn=False)

    last_frame_xyz = l1_xyz[:, (NUM_INPUT_FRAMES - 1) * NUM_PARTICLES:, :]

    l2_xyz, l2_points, l2_indices = pointnet_sa_module(last_frame_xyz,
                                                       l1_points,
                                                       npoint=NUM_PARTICLES,
                                                       radius=0.3,
                                                       nsample=NUM_PARTICLES,
                                                       mlp=[128, 128, 256],
                                                       group_all=False)

    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz,
                                                       l2_points,
                                                       npoint=NUM_PARTICLES,
                                                       radius=0.3,
                                                       nsample=NUM_PARTICLES,
                                                       mlp=[256, 256, 512],
                                                       group_all=False)

    # =========================== Feature Propagation =====================

    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256])
    l1_points = pointnet_fp_module(l1_xyz[:, (NUM_INPUT_FRAMES - 1) * NUM_PARTICLES:, :], l2_xyz, l1_points[:, (NUM_INPUT_FRAMES - 1) * NUM_PARTICLES:, :], l2_points, [256,256])
    l0_points = pointnet_fp_module(l0_xyz[:, (NUM_INPUT_FRAMES - 1) * NUM_PARTICLES:, :], l1_xyz, l0_points[:, (NUM_INPUT_FRAMES - 1) * NUM_PARTICLES:, :], l1_points, [128, 128, 3], last=True)

    predicted_point_cloud = Reshape((NUM_PARTICLES, 3), name='predicted_point_cloud')(l0_points)
    meteor_net_model = Model(inputs=meteor_net_input, outputs=predicted_point_cloud)
    return meteor_net_model
