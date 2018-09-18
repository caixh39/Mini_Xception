import os
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, UpSampling2D
from keras.layers import AveragePooling2D, BatchNormalization, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers import Reshape, Dense, multiply, Permute
from keras.regularizers import l2
from keras import layers
from keras import backend as K
from keras.utils import plot_model
from Module_Net import squeeze_excite_block, conv2d_bn, sep_conv2d_bn


# author_training_aug_disgusted: 66.98%
def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output)
    return model


# 67.82% , 68.07%   Vs. mini_Xception (concated instead of add)
def mini_concate_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([residual, x], name='mixed1')
    x = squeeze_excite_block(x, ratio=16)

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([residual, x], name='mixed2')
    x = squeeze_excite_block(x, ratio=16)

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.concatenate([residual, x], name='mixed3')
    x = squeeze_excite_block(x, ratio=16)

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.concatenate([residual, x], name='mixed4')
    x = squeeze_excite_block(x, ratio=16)

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output)
    return model


# Vs. mini_concate_XCEPTION, adding sep(1x1+3x3)
def mini_concate_V2_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    branch1_1 = Conv2D(16, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x)
    branch1_1 = BatchNormalization()(branch1_1)

    branch1_2 = SeparableConv2D(16, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch1_2 = BatchNormalization()(branch1_2)
    branch1_2 = Activation('relu')(branch1_2)
    branch1_2 = SeparableConv2D(16, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch1_2)
    branch1_2 = BatchNormalization()(branch1_2)

    branch1_3 = SeparableConv2D(16, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch1_3 = BatchNormalization()(branch1_3)
    branch1_3 = Activation('relu')(branch1_3)
    branch1_3 = SeparableConv2D(16, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch1_3)
    branch1_3 = BatchNormalization()(branch1_3)
    branch1_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_3)

    x = layers.concatenate([branch1_1, branch1_2, branch1_3], name='mixed1')
    x = squeeze_excite_block(x, ratio=16)

    # module 2
    branch2_1 = Conv2D(32, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x)
    branch2_1 = BatchNormalization()(branch2_1)

    branch2_2 = SeparableConv2D(32, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch2_2 = BatchNormalization()(branch2_2)
    branch2_2 = Activation('relu')(branch2_2)
    branch2_2 = SeparableConv2D(32, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch2_2)
    branch2_2 = BatchNormalization()(branch2_2)

    branch2_3 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch2_3 = BatchNormalization()(branch2_3)
    branch2_3 = Activation('relu')(branch2_3)
    branch2_3 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch2_3)
    branch2_3 = BatchNormalization()(branch2_3)
    branch2_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_3)

    x = layers.concatenate([branch2_1, branch2_2, branch2_3], name='mixed2')
    x = squeeze_excite_block(x, ratio=16)

    # module 3
    branch3_1 = Conv2D(64, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x)
    branch3_1 = BatchNormalization()(branch3_1)

    branch3_2 = SeparableConv2D(64, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch3_2 = BatchNormalization()(branch3_2)
    branch3_2 = Activation('relu')(branch3_2)
    branch3_2 = SeparableConv2D(64, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch3_2 = BatchNormalization()(branch3_2)

    branch3_3 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch3_3 = BatchNormalization()(branch3_3)
    branch3_3 = Activation('relu')(branch3_3)
    branch3_3 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch3_3)
    branch3_3 = BatchNormalization()(branch3_3)
    branch3_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_3)

    x = layers.concatenate([branch3_1, branch3_2, branch3_3], name='mixed3')
    x = squeeze_excite_block(x, ratio=16)

    # module 4
    branch4_1 = Conv2D(128, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x)
    branch4_1 = BatchNormalization()(branch4_1)

    branch4_2 = SeparableConv2D(128, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch4_2 = BatchNormalization()(branch4_2)
    branch4_2 = Activation('relu')(branch4_2)
    branch4_2 = SeparableConv2D(128, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch4_2)
    branch4_2 = BatchNormalization()(branch4_2)

    branch4_3 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch4_3 = BatchNormalization()(branch4_3)
    branch4_3 = Activation('relu')(branch4_3)
    branch4_3 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch4_3)
    branch4_3 = BatchNormalization()(branch4_3)
    branch4_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_3)

    x = layers.concatenate([branch4_1, branch4_2, branch4_3], name='mixed4')
    x = squeeze_excite_block(x, ratio=16)

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output)
    return model


def mini_concate_V3_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    branch1_1 = Conv2D(16, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x)
    branch1_1 = BatchNormalization()(branch1_1)

    branch1_2 = SeparableConv2D(16, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch1_2 = BatchNormalization()(branch1_2)
    branch1_2 = Activation('relu')(branch1_2)
    branch1_2 = SeparableConv2D(16, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch1_2)
    branch1_2 = BatchNormalization()(branch1_2)

    branch1_3 = SeparableConv2D(16, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch1_3 = BatchNormalization()(branch1_3)
    branch1_3 = Activation('relu')(branch1_3)
    branch1_3 = SeparableConv2D(16, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch1_3)
    branch1_3 = BatchNormalization()(branch1_3)
    branch1_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_3)

    x_1 = layers.concatenate([branch1_1, branch1_2, branch1_3], name='mixed1')
    x_1 = squeeze_excite_block(x_1, ratio=16)

    # module 2
    branch2_1 = Conv2D(32, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x_1)
    branch2_1 = BatchNormalization()(branch2_1)

    branch2_2 = SeparableConv2D(32, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x_1)
    branch2_2 = BatchNormalization()(branch2_2)
    branch2_2 = Activation('relu')(branch2_2)
    branch2_2 = SeparableConv2D(32, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch2_2)
    branch2_2 = BatchNormalization()(branch2_2)

    branch2_3 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x_1)
    branch2_3 = BatchNormalization()(branch2_3)
    branch2_3 = Activation('relu')(branch2_3)
    branch2_3 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch2_3)
    branch2_3 = BatchNormalization()(branch2_3)
    branch2_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_3)

    x_2 = layers.concatenate([branch2_1, branch2_2, branch2_3], name='mixed2')
    x_2 = squeeze_excite_block(x_2, ratio=16)

    # module 3
    branch3_1 = Conv2D(64, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x_2)
    branch3_1 = BatchNormalization()(branch3_1)

    branch3_2 = SeparableConv2D(64, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x_2)
    branch3_2 = BatchNormalization()(branch3_2)
    branch3_2 = Activation('relu')(branch3_2)
    branch3_2 = SeparableConv2D(64, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch3_2)
    branch3_2 = BatchNormalization()(branch3_2)

    branch3_3 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x_2)
    branch3_3 = BatchNormalization()(branch3_3)
    branch3_3 = Activation('relu')(branch3_3)
    branch3_3 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch3_3)
    branch3_3 = BatchNormalization()(branch3_3)
    branch3_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_3)

    x_3 = layers.concatenate([branch3_1, branch3_2, branch3_3], name='mixed3')
    x_3 = squeeze_excite_block(x_3, ratio=16)

    # module 4
    branch4_1 = Conv2D(128, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x_3)
    branch4_1 = BatchNormalization()(branch4_1)

    branch4_2 = SeparableConv2D(128, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x_3)
    branch4_2 = BatchNormalization()(branch4_2)
    branch4_2 = Activation('relu')(branch4_2)
    branch4_2 = SeparableConv2D(128, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch4_2)
    branch4_2 = BatchNormalization()(branch4_2)

    branch4_3 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x_3)
    branch4_3 = BatchNormalization()(branch4_3)
    branch4_3 = Activation('relu')(branch4_3)
    branch4_3 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch4_3)
    branch4_3 = BatchNormalization()(branch4_3)
    branch4_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_3)

    x_4 = layers.concatenate([branch4_1, branch4_2, branch4_3], name='mixed4')
    x_4 = squeeze_excite_block(x_4, ratio=16)

    x_2 = GlobalAveragePooling2D()(x_2)
    x_3 = GlobalAveragePooling2D()(x_3)
    x_4 = GlobalAveragePooling2D()(x_4)

    # x_2 = GlobalMaxPooling2D()(x_2)
    # x_3 = GlobalMaxPooling2D()(x_3)
    # x_4 = GlobalMaxPooling2D()(x_4)

    x = layers.concatenate([x_2, x_3, x_4], name='layer_concate')
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output)
    return model


def InceptionV4_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)
    img_input = Input(input_shape)

    # base
    x = sep_conv2d_bn(img_input, 8, 3, 3, padding='valid')
    x = sep_conv2d_bn(x, 16, 3, 3, padding='valid')

    # multi scale
    x_01 = sep_conv2d_bn(x, 16, 3, 3, padding='valid', strides=(2, 2))
    x_02 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x_02 = BatchNormalization()(x_02)

    x = layers.concatenate([x_01, x_02], name='mixed0')
    x = squeeze_excite_block(x, ratio=16)

    # Inception-ResNet-v2: module A
    branch1_1 = conv2d_bn(x, 16, 1, 1)

    branch1_2 = sep_conv2d_bn(x, 8, 1, 1)
    branch1_2 = sep_conv2d_bn(branch1_2, 8, 3, 3)

    branch1_3 = sep_conv2d_bn(x, 8, 1, 1)
    branch1_3 = sep_conv2d_bn(branch1_3, 16, 3, 3)
    branch1_3 = sep_conv2d_bn(branch1_3, 16, 3, 3)

    branches = [branch1_1, branch1_2, branch1_3]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_1')(branches)
    x_1 = squeeze_excite_block(x, ratio=16)

    # Inception-ResNet-v2: reduce_1.1
    branch1_r1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x_1)

    branch1_r2 = sep_conv2d_bn(x_1, 16, 1, 1)
    branch1_r2 = sep_conv2d_bn(branch1_r2, 32, 3, 3, strides=(2, 2), padding='valid')

    branch1_r3 = sep_conv2d_bn(x_1, 16, 1, 1)
    branch1_r3 = sep_conv2d_bn(branch1_r3, 16, 3, 3)
    branch1_r3 = sep_conv2d_bn(branch1_r3, 32, 3, 3, strides=(2, 2), padding='valid')

    branches = [branch1_r1, branch1_r2, branch1_r3]
    x_r1 = layers.Concatenate(axis=channel_axis, name='mixed_reduce1')(branches)
    x_r1 = squeeze_excite_block(x_r1, ratio=16)

    # Inception-ResNet-v2: module B_1
    branch2_1 = conv2d_bn(x_r1, 64, 1, 1)

    branch2_2 = sep_conv2d_bn(x_r1, 48, 1, 1)
    branch2_2 = sep_conv2d_bn(branch2_2, 48, 1, 3)
    branch2_2 = sep_conv2d_bn(branch2_2, 64, 3, 1)

    branches = [branch2_1, branch2_2]
    x = layers.Concatenate(axis=channel_axis, name='mixed_2')(branches)
    x_2 = squeeze_excite_block(x, ratio=16)

    branch2_1 = conv2d_bn(x_2, 64, 1, 1)

    branch2_2 = sep_conv2d_bn(x_2, 48, 1, 1)
    branch2_2 = sep_conv2d_bn(branch2_2, 48, 1, 7)
    branch2_2 = sep_conv2d_bn(branch2_2, 64, 7, 1)

    branches = [branch2_1, branch2_2]
    x = layers.Concatenate(axis=channel_axis, name='mixed_21')(branches)
    x_2 = squeeze_excite_block(x, ratio=16)

    # reduce_2.1 module
    branch2_r1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x_2)

    branch2_r2 = conv2d_bn(x_2, 96, 1, 1)
    branch2_r2 = sep_conv2d_bn(branch2_r2, 128, 3, 3, strides=(2, 2), padding='valid')

    branch2_r3 = sep_conv2d_bn(x_2, 64, 1, 1)
    branch2_r3 = sep_conv2d_bn(branch2_r3, 96, 3, 3, strides=(2, 2), padding='valid')

    branch2_r4 = sep_conv2d_bn(x_2, 64, 1, 1)
    branch2_r4 = sep_conv2d_bn(branch2_r4, 96, 3, 3)
    branch2_r4 = sep_conv2d_bn(branch2_r4, 128, 3, 3, strides=(2, 2), padding='valid')

    branches = [branch2_r1, branch2_r2, branch2_r3, branch2_r4]
    x_r3 = layers.Concatenate(axis=channel_axis, name='mixed_reduce3')(branches)
    x_r3 = squeeze_excite_block(x_r3, ratio=16)

    # module C
    branch3_1 = conv2d_bn(x_r3, 64, 1, 1)

    branch3_2 = sep_conv2d_bn(x_r3, 64, 1, 1)
    branch3_2 = sep_conv2d_bn(branch3_2, 96, 1, 3)
    branch3_2 = sep_conv2d_bn(branch3_2, 128, 3, 3)

    branches = [branch3_1, branch3_2]
    x = layers.Concatenate(axis=channel_axis, name='mixed_4')(branches)
    x_3 = squeeze_excite_block(x, ratio=16)

    x_2 = GlobalAveragePooling2D(name='avg_pool_1')(x_2)
    x_3 = GlobalAveragePooling2D(name='avg_pool_2')(x_3)
    x_r3 = GlobalAveragePooling2D(name='avg_pool_3')(x_r3)

    x = layers.concatenate([x_2, x_3, x_r3], name='layer_concate')

    x = Dropout(0.5)(x)
    # print x_3.shape
    # os._exit(0)
    # x = sep_conv2d_bn(x, num_classes, 3, 3)

    output = Dense(num_classes, activation='softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output)
    return model


if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    # model = mini_XCEPTION(input_shape, num_classes)
    # model.summary() 
    # model = mini_concate_XCEPTION(input_shape, num_classes)
    # # model.summary()
    model = InceptionV4_XCEPTION(input_shape, num_classes)
    model.summary()
    # model = plot_model(model, to_file='../../images/models/InceptionV4_XCEPTION_0914.png', show_shapes=True,
    #                    show_layer_names=True)
