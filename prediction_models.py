from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Input, Reshape, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LayerNormalization, AveragePooling2D
import numpy as np
from loss import *


def model_builder(args):
    basic_model = build_timewise_pointnet(args)
    model = build_recursive_prediction_model(args, basic_model)

    if args.loss_fn == 'cd':
        first_loss = get_cd_loss_func_for_first
        base_loss = get_cd_loss_func
    elif args.loss_fn == 'advanced_cd':
        first_loss = get_advanced_cd_loss_func_for_first
        base_loss = get_advanced_cd_loss_func
    else:
        first_loss = mse_for_first
        base_loss = mse_base

    loss = {f'tf_op_layer_output0': first_loss}
    for i in range(1, args.num_output):
        loss[f'tf_op_layer_output{i}'] = base_loss

    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr,
                                                     beta_1=args.beta_1,
                                                     beta_2=args.beta_2,
                                                     epsilon=args.epsilon,
                                                     amsgrad=bool(args.use_amsgrad)))
    model.summary()
    return model


def input_transform_net(point_cloud, K):

    num_point = point_cloud.get_shape()[1]

    input_image = tf.expand_dims(point_cloud, -1)

    net = Conv2D(filters=64,
                 kernel_size=(1, K),
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

        weights = tf.compat.v1.get_variable('weights', [128, K*K],
                                            initializer=tf.constant_initializer(0.0),
                                            dtype=tf.float32)

        biases = tf.compat.v1.get_variable('biases', [K*K],
                                           initializer=tf.constant_initializer(0.0),
                                           dtype=tf.float32)

        biases.assign_add(tf.constant([1, 0, 0, 1], dtype=tf.float32))

        transform = tf.matmul(net, weights)

        transform = tf.nn.bias_add(transform, biases)

    transform = Reshape(target_shape=(K, K))(transform)

    return transform


def feature_transform_net(inputs, K=64):

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


def build_shared_feature_extractor(args):

    input_point_set = Input(shape=(args.n_particles, args.n_dims))

    if args.use_transform_net:
        with tf.compat.v1.variable_scope('transform_net1') as sc:
            transform = input_transform_net(input_point_set, K=args.n_dims)
        point_cloud_transformed = tf.matmul(input_point_set, transform)
        net = tf.expand_dims(point_cloud_transformed, -1)
    else:
        net = tf.expand_dims(input_point_set, -1)

    net = Conv2D(filters=64,
                 kernel_size=(1, args.n_dims),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    if args.bn:
        net = BatchNormalization()(net)

    net = Conv2D(filters=64,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    if args.bn:
        net = BatchNormalization()(net)

    if args.use_transform_net:
        with tf.compat.v1.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, K=64)
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net = tf.expand_dims(net_transformed, [2])

    net = Conv2D(filters=128,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    if args.bn:
        net = BatchNormalization()(net)

    net = Conv2D(filters=256,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    if args.bn:
        net = BatchNormalization()(net)

    net = Conv2D(filters=512,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 activation=tf.nn.relu,
                 padding='valid',
                 data_format='channels_last')(net)

    if args.bn:
        net = BatchNormalization()(net)

    if args.pool_type == 'max':
        net = MaxPooling2D(pool_size=(args.n_particles, 1), padding='valid')(net)
    else:
        net = AveragePooling2D(pool_size=(args.n_particles, 1), padding='valid')(net)

    net = Flatten()(net)
    output = Dense(args.n_global_features, activation=tf.nn.relu)(net)

    shared_feature_extractor = Model(inputs=input_point_set, outputs=output)
    shared_feature_extractor.summary()

    return shared_feature_extractor


def build_timewise_pointnet(args):

    model_input = Input(shape=(args.num_input, args.n_particles, args.n_dims))

    shared_feature_extractor = build_shared_feature_extractor(args)

    nth_features = []

    for i in range(args.num_input):
        features = tf.expand_dims(shared_feature_extractor(model_input[:, i]), axis=1)
        nth_features.append(features)

    features_per_timestep = tf.concat((nth_features[0], nth_features[1]), axis=1)
    for i in range(2, args.num_input):
        features_per_timestep = tf.concat((features_per_timestep, nth_features[i]), axis=1)

    net = LSTM(units=512,
               activation=tf.nn.relu,
               return_sequences=True)(features_per_timestep)

    net = LayerNormalization()(net)

    net = LSTM(units=256,
               activation=tf.nn.relu,
               return_sequences=True)(net)

    net = LayerNormalization()(net)

    net = LSTM(units=256,
               activation=tf.nn.relu,
               return_sequences=False)(net)

    net = LayerNormalization()(net)

    net = Dense(units=args.n_particles * args.n_dims,
                activation=tf.nn.sigmoid)(net)

    prediction_output = Reshape((args.n_particles, args.n_dims))(net)

    model = Model(inputs=model_input, outputs=prediction_output)
    model.summary()
    return model


def build_recursive_prediction_model(args, single_frame_prediction_model):

    """ Multiple Frames Prediction Model
        This model iteratively uses Single Frame Prediction Model to recursively
        predict frames at timestep t+1, t+2, ... , t+8. """

    outputs = []
    initial_input = Input(shape=(args.num_input, args.n_particles, args.n_dims))

    model_input = initial_input
    output = single_frame_prediction_model(model_input)
    outputs.append(output)

    for i in range(args.num_output - 1):
        model_input = tf.concat((model_input[:, 1:args.num_input], tf.expand_dims(outputs[i], axis=1)), axis=1)
        output = single_frame_prediction_model(model_input)
        outputs.append(output)

    outputs = [tf.identity(outputs[i], name=f'output{i}') for i in range(args.num_output)]
    recursive_prediction_model = Model(inputs=initial_input, outputs=outputs)
    return recursive_prediction_model
