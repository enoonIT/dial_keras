from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from convnetskeras.customlayers import crosschannelnormalization, splittensor
from keras import backend as K
from multibn import multibn_block


def get_alexnet_shape():
    dim_ordering = K.image_dim_ordering()
    default_size = 227
    if dim_ordering == 'th':
        default_shape = (3, default_size, default_size)
    else:
        default_shape = (default_size, default_size, 3)
    return default_shape


def vgg_like(n_classes=1000, multibn_layer=False):
    dim_ordering = K.image_dim_ordering()
    default_size = 224
    if dim_ordering == 'th':
        default_shape = (3, default_size, default_size)
    else:
        default_shape = (default_size, default_size, 3)

    input = Input(shape=default_shape)
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    num = 1
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    if multibn_layer:
        x = multibn_block(x, num)
    x = Dense(4096, activation='relu', name='fc2')(x)
    if multibn_layer:
        x = multibn_block(x, num)
    x = Dense(n_classes, activation='softmax', name='predictions')(x)
    return Model(input=input, output=x)


def AlexNet(weights_path=None, n_classes=1000, multibn=False):
    inputs = Input(shape=get_alexnet_shape())
    conv_1 = Convolution2D(
        96, 11, 11, subsample=(4, 4), activation='relu', name='conv_1')(inputs)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge(
        [
            Convolution2D(
                128, 5, 5, activation="relu",
                name='conv_2_' + str(i + 1))(splittensor(
                    ratio_split=2, id_split=i)(conv_2)) for i in range(2)
        ],
        mode='concat',
        concat_axis=1,
        name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge(
        [
            Convolution2D(
                192, 3, 3, activation="relu",
                name='conv_4_' + str(i + 1))(splittensor(
                    ratio_split=2, id_split=i)(conv_4)) for i in range(2)
        ],
        mode='concat',
        concat_axis=1,
        name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge(
        [
            Convolution2D(
                128, 3, 3, activation="relu",
                name='conv_5_' + str(i + 1))(splittensor(
                    ratio_split=2, id_split=i)(conv_5)) for i in range(2)
        ],
        mode='concat',
        concat_axis=1,
        name="conv_5")

    num = 1
    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)
    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    if multibn:
        dense_1 = multibn_block(dense_1, num, name="MultiBN1")
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    if multibn:
        dense_2 = multibn_block(dense_2, num, name="MultiBN2")
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(n_classes, name='dense_3')(dense_3)
    prediction = Activation("softmax", name="softmax")(dense_3)

    model = Model(input=inputs, output=prediction)
    if weights_path:
        model.load_weights(weights_path)
    return model
