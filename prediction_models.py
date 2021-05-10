from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Input, Reshape, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LayerNormalization
from common.Constants import *
import tensorflow as tf
import numpy as np


def simple_lstm():

    prediction_input = Input(shape=(NUM_INPUT_FRAMES, NUM_PARTICLES, 2))

    net = Reshape((NUM_INPUT_FRAMES, NUM_PARTICLES * 2))(prediction_input)

    net = Bidirectional(LSTM(units=256,
                             return_sequences=True,
                             activation=tf.nn.relu))(net)

    net = LayerNormalization()(net)

    net = Bidirectional(LSTM(units=128,
                             return_sequences=True,
                             activation=tf.nn.relu))(net)

    net = LayerNormalization()(net)

    net = Bidirectional(LSTM(units=64,
                             return_sequences=False,
                             activation=tf.nn.relu))(net)

    net = LayerNormalization()(net)

    net = Dense(units=NUM_PARTICLES * 2,
                activation=tf.nn.sigmoid)(net)

    prediction_output = Reshape((NUM_PARTICLES, 2))(net)

    prediction_model = Model(inputs=prediction_input, outputs=prediction_output)
    prediction_model.summary()

    return prediction_model


def input_transform_net(point_cloud, K=2):
    """ Input (XY) Transform Net, input is BxNx2 gray image
        Return:
            Transformation matrix of size 2xK """

    num_point = point_cloud.get_shape()[1]

    input_image = tf.expand_dims(point_cloud, -1)

    net = Conv2D(filters=64,
                 kernel_size=(1, 2),
                 activation=tf.nn.relu,
                 padding='valid',
                 strides=(1, 1))(input_image)

    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 strides=(1, 1))(net)

    net = Conv2D(filters=512,
                 kernel_size=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 strides=(1, 1))(net)

    net = MaxPooling2D(pool_size=(num_point, 1),
                       padding='valid')(net)

    net = Reshape(target_shape=(-1, ))(net)

    net = Dense(256, activation=tf.nn.relu)(net)
    net = Dense(128, activation=tf.nn.relu)(net)

    with tf.compat.v1.variable_scope('transform_XY2') as sc:
        assert(K == 2)
        weights = tf.compat.v1.get_variable('weights', [128, 2*K],
                                            initializer=tf.constant_initializer(0.0),
                                            dtype=tf.float32)

        biases = tf.compat.v1.get_variable('biases', [2*K],
                                           initializer=tf.constant_initializer(0.0),
                                           dtype=tf.float32)

        biases.assign_add(tf.constant([1, 0, 0, 1], dtype=tf.float32))

        transform = tf.matmul(net, weights)

        transform = tf.nn.bias_add(transform, biases)

    transform = Reshape(target_shape=(2, K))(transform)

    return transform


def feature_transform_net(inputs, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """

    num_point = inputs.get_shape()[1]

    net = Conv2D(filters=64,
                 kernel_size=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 strides=(1, 1))(inputs)

    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 strides=(1, 1))(net)

    net = Conv2D(filters=512,
                 kernel_size=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 strides=(1, 1))(net)

    net = MaxPooling2D(pool_size=(num_point, 1),
                       padding='valid')(net)

    net = Reshape(target_shape=(-1, ))(net)

    net = Dense(256, activation=tf.nn.relu)(net)
    net = Dense(128, activation=tf.nn.relu)(net)

    with tf.compat.v1.variable_scope('transform_feat') as sc:
        weights = tf.compat.v1.get_variable('weights', [128, K*K],
                                            initializer=tf.constant_initializer(0.0),
                                            dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [K*K],
                                           initializer=tf.constant_initializer(0.0),
                                           dtype=tf.float32)
        biases.assign_add(tf.constant(np.eye(K).flatten(), dtype=tf.float32))

        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = Reshape(target_shape=(K, K))(transform)

    return transform


def build_global_feature_extractor(num_global_features, use_transform_net, bn):

    """ Variation of PointNet Model which extracts global features from a 2D PointSet """

    point_set_input = Input(shape=(NUM_PARTICLES, 2))

    if use_transform_net:
        with tf.compat.v1.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_set_input, K=2)
        point_cloud_transformed = tf.matmul(point_set_input, transform)

        net = tf.expand_dims(point_cloud_transformed, -1)
    else:
        net = tf.expand_dims(point_set_input, -1)

    net1 = Conv2D(filters=64,
                 kernel_size=(1, 2),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    if bn:
        net1 = BatchNormalization()(net1)

    net2 = Conv2D(filters=64,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net1)

    if bn:
        net2 = BatchNormalization()(net2)

    if use_transform_net:
        with tf.compat.v1.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net1 + net2, K=64)
        net_transformed = tf.matmul(tf.squeeze(net1 + net2, axis=[2]), transform)
        net2 = tf.expand_dims(net_transformed, [2])

    # Changed from 128 -> 256 for skip connections
    net3 = Conv2D(filters=256,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net2)

    if bn:
        net3 = BatchNormalization()(net3)

    net4 = Conv2D(filters=256,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net3)

    if bn:
        net4 = BatchNormalization()(net4)

    net = Conv2D(filters=512,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net3 + net4)

    if bn:
        net = BatchNormalization()(net)

    net = MaxPooling2D(pool_size=(NUM_PARTICLES, 1), padding='valid')(net)

    net = Flatten()(net)

    output = Dense(num_global_features, activation=tf.nn.relu)(net)

    global_feature_extractor = Model(inputs=point_set_input, outputs=output)
    global_feature_extractor.summary()

    return global_feature_extractor


def global_pointnet_lstm(num_global_features=128, use_transform_net=True, bn=False):

    prediction_input = Input(shape=(NUM_INPUT_FRAMES, NUM_PARTICLES, 2))

    global_feature_extractor = build_global_feature_extractor(num_global_features, use_transform_net, bn)

    nth_features = []

    for i in range(NUM_INPUT_FRAMES):
        features = tf.expand_dims(global_feature_extractor(prediction_input[:, i]), axis=1)
        nth_features.append(features)

    features_per_timestep = tf.concat((nth_features[0], nth_features[1]), axis=1)
    for i in range(2, NUM_INPUT_FRAMES):
        features_per_timestep = tf.concat((features_per_timestep, nth_features[i]), axis=1)

    net = Bidirectional(LSTM(units=256, return_sequences=True))(features_per_timestep)

    net = LayerNormalization()(net)

    net = Bidirectional(LSTM(units=128,
                             return_sequences=True))(net)

    net = LayerNormalization()(net)

    net = Bidirectional(LSTM(units=128,
                             return_sequences=False))(net)

    net = LayerNormalization()(net)

    net = Dense(units=NUM_PARTICLES * 2,
                activation=tf.nn.sigmoid)(net)

    prediction_output = Reshape((NUM_PARTICLES, 2))(net)

    prediction_model = Model(inputs=prediction_input, outputs=prediction_output)
    prediction_model.summary()
    return prediction_model


def build_local_global_feature_extractor(num_local_features, num_global_features, bn):

    """ Variation of PointNet Model which extracts global features from a 2D PointSet """

    point_set_input = Input(shape=(NUM_PARTICLES, 2))
    net = tf.expand_dims(point_set_input, axis=-1)

    net = Conv2D(filters=64,
                 kernel_size=(1, 2),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    if bn:
        net = BatchNormalization()(net)

    local_features = Conv2D(filters=num_local_features,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation=tf.nn.relu,
                            padding='valid',
                            data_format='channels_last')(net)

    if bn:
        local_features = BatchNormalization()(local_features)

    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(local_features)

    if bn:
        net = BatchNormalization()(net)

    net = Conv2D(filters=256,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    if bn:
        net = BatchNormalization()(net)

    net = Conv2D(filters=512,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    if bn:
        net = BatchNormalization()(net)

    net = MaxPooling2D(pool_size=(NUM_PARTICLES, 1), padding='valid')(net)

    net = Flatten()(net)

    global_features = Dense(num_global_features, activation=tf.nn.relu)(net)

    local_global_feature_extractor = Model(inputs=point_set_input, outputs=[local_features, global_features])
    local_global_feature_extractor.summary()

    return local_global_feature_extractor


def pointnet_lstm(num_local_features=64, num_global_features=128, bn=False):

    prediction_input = Input(shape=(NUM_INPUT_FRAMES, NUM_PARTICLES, 2))

    per_point_feature_extractor = build_local_global_feature_extractor(num_local_features, num_global_features, bn)

    _, first_global_features = per_point_feature_extractor(prediction_input[:, 0])
    _, second_global_features = per_point_feature_extractor(prediction_input[:, 1])
    third_local_features, third_global_features = per_point_feature_extractor(prediction_input[:, 2])

    first_global_features = Reshape(target_shape=(num_global_features, ))(first_global_features)
    first_global_features = tf.tile(first_global_features, [1, NUM_PARTICLES])
    first_global_features = Reshape(target_shape=(NUM_PARTICLES, 1, num_global_features))(first_global_features)

    second_global_features = Reshape(target_shape=(num_global_features, ))(second_global_features)
    second_global_features = tf.tile(second_global_features, [1, NUM_PARTICLES])
    second_global_features = Reshape(target_shape=(NUM_PARTICLES, 1, num_global_features))(second_global_features)

    third_global_features = Reshape(target_shape=(num_global_features, ))(third_global_features)
    third_global_features = tf.tile(third_global_features, [1, NUM_PARTICLES])
    third_global_features = Reshape(target_shape=(NUM_PARTICLES, 1, num_global_features))(third_global_features)

    total_features_per_timestep = tf.concat((first_global_features, second_global_features, third_local_features, third_global_features), axis=3)

    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(total_features_per_timestep)

    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    net = Conv2D(filters=2,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.sigmoid,
                 padding='valid',
                 data_format='channels_last')(net)

    prediction_output = Reshape((NUM_PARTICLES, 2))(net)

    prediction_model = Model(inputs=prediction_input, outputs=prediction_output)
    prediction_model.summary()

    return prediction_model


def build_recursive_prediction_model(single_frame_prediction_model):

    """ Multiple Frames Prediction Model
        This model iteratively uses Single Frame Prediction Model to recursively
        predict frames at timestep t+1, t+2, ... , t+8. """

    first_input = Input(shape=(NUM_INPUT_FRAMES, NUM_PARTICLES, 2))
    output1 = single_frame_prediction_model(first_input)

    second_input = tf.concat((first_input[:, 1:NUM_INPUT_FRAMES], tf.expand_dims(output1, axis=1)), axis=1)
    output2 = single_frame_prediction_model(second_input)

    third_input = tf.concat((second_input[:, 1:NUM_INPUT_FRAMES], tf.expand_dims(output2, axis=1)), axis=1)
    output3 = single_frame_prediction_model(third_input)

    fourth_input = tf.concat((third_input[:, 1:NUM_INPUT_FRAMES], tf.expand_dims(output3, axis=1)), axis=1)
    output4 = single_frame_prediction_model(fourth_input)

    fifth_input = tf.concat((fourth_input[:, 1:NUM_INPUT_FRAMES], tf.expand_dims(output4, axis=1)), axis=1)
    output5 = single_frame_prediction_model(fifth_input)

    sixth_input = tf.concat((fifth_input[:, 1:NUM_INPUT_FRAMES], tf.expand_dims(output5, axis=1)), axis=1)
    output6 = single_frame_prediction_model(sixth_input)

    seventh_input = tf.concat((sixth_input[:, 1:NUM_INPUT_FRAMES], tf.expand_dims(output6, axis=1)), axis=1)
    output7 = single_frame_prediction_model(seventh_input)

    eighth_input = tf.concat((seventh_input[:, 1:NUM_INPUT_FRAMES], tf.expand_dims(output7, axis=1)), axis=1)
    output8 = single_frame_prediction_model(eighth_input)

    output1 = tf.identity(output1, name='output1')
    output2 = tf.identity(output2, name='output2')
    output3 = tf.identity(output3, name='output3')
    output4 = tf.identity(output4, name='output4')
    output5 = tf.identity(output5, name='output5')
    output6 = tf.identity(output6, name='output6')
    output7 = tf.identity(output7, name='output7')
    output8 = tf.identity(output8, name='output8')

    recursive_prediction_model = Model(inputs=first_input, outputs=[output1, output2, output3, output4, output5, output6, output7,
                                                                    output8])
    return recursive_prediction_model


if __name__ == '__main__':
    import numpy as np
    input_point_set = np.zeros(shape=(32, NUM_INPUT_FRAMES, 20, 2))
    model = build_recursive_prediction_model(global_pointnet_lstm(use_transform_net=True))
    model.summary()
