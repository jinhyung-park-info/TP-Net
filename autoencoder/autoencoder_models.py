from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Dropout, UpSampling2D, MaxPooling2D, Conv2D
import tensorflow as tf
from common.Constants import NUM_PARTICLES


def build_encoder(activation_function):

    grey_image = Input(shape=(480, 480))
    net = tf.expand_dims(grey_image, axis=-1)

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

    net = Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = MaxPooling2D(pool_size=(4, 4),
                       data_format='channels_last')(net)

    net = Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = Conv2D(filters=128,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=tf.nn.leaky_relu,
                 padding='same',
                 data_format='channels_last')(net)

    net = MaxPooling2D(pool_size=(4, 4),
                       data_format='channels_last')(net)

    net = Conv2D(filters=64,
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

    net = MaxPooling2D(pool_size=(2, 2),
                       data_format='channels_last')(net)

    net = Flatten()(net)
    net = Dropout(0.4)(net)

    if activation_function == 'relu':
        net = Dense(NUM_PARTICLES * 2, activation=tf.nn.sigmoid)(net)
    else:
        net = Dense(NUM_PARTICLES * 2, activation=tf.nn.tanh)(net)

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
