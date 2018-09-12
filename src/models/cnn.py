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
from Module_Net import squeeze_excite_block, conv2d_bn


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


# 67.68%
def GAP_concate_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = conv2d_bn(img_input, 16, 3, 3, strides=(1, 1),
                  kernel_regularizer=regularization, padding='valid')
    x_0 = conv2d_bn(x, 16, 3, 3, strides=(1, 1),
                    kernel_regularizer=regularization, padding='valid')

    # x_00 = Conv2D(num_classes, (5, 5), strides=(1, 1), padding='valid',
    #             kernel_regularizer=regularization)(x_0)


    # module 1 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch1_11 = conv2d_bn(x_0, 16, 1, 1, strides=(1, 1),
                           kernel_regularizer=regularization, padding='same')
    branch1_12 = SeparableConv2D(32, (3, 3), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch1_11)
    branch1_12 = BatchNormalization()(branch1_12)
    branch1_12 = Activation('relu')(branch1_12)
    branch1_13 = SeparableConv2D(32, (3, 3), strides=(1, 1), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch1_12)
    branch1_13 = BatchNormalization()(branch1_13)
    branch1_13 = Activation('relu')(branch1_13)
    branch1_13 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_13)

    branch1_21 = conv2d_bn(x_0, 16, 1, 1, strides=(1, 1), padding='same')
    branch1_22 = SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch1_21)
    branch1_22 = BatchNormalization()(branch1_22)
    branch1_22 = Activation('relu')(branch1_22)
    branch1_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_22)

    branch1_41 = SeparableConv2D(16, (3, 3), strides=(2, 2), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(x_0)
    branch1_41 = BatchNormalization()(branch1_41)
    branch1_41 = Activation('relu')(branch1_41)

    x_1 = layers.concatenate([branch1_13, branch1_22, branch1_41], name='mixed1')
    x_1 = squeeze_excite_block(x_1, ratio=16)

    x_01 = GlobalAveragePooling2D()(x_1)

    # module 2 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch2_11 = conv2d_bn(x_1, 32, 1, 1, strides=(1, 1),
                           kernel_regularizer=regularization, padding='same')
    branch2_12 = SeparableConv2D(64, (3, 3), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch2_11)
    branch2_12 = BatchNormalization()(branch2_12)
    branch2_12 = Activation('relu')(branch2_12)
    branch2_13 = SeparableConv2D(64, (3, 3), strides=(1, 1), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch2_12)
    branch2_13 = BatchNormalization()(branch2_13)
    branch2_13 = Activation('relu')(branch2_13)
    branch2_13 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_13)

    branch2_21 = conv2d_bn(x_1, 32, 1, 1, strides=(1, 1), padding='same')
    branch2_22 = SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch2_21)
    branch2_22 = BatchNormalization()(branch2_22)
    branch2_22 = Activation('relu')(branch2_22)
    branch2_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_22)

    branch2_41 = SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(x_1)
    branch2_41 = BatchNormalization()(branch2_41)
    branch2_41 = Activation('relu')(branch2_41)

    x_2 = layers.concatenate([branch2_13, branch2_22, branch2_41],
                             name='mixed2')
    x_2 = squeeze_excite_block(x_2, ratio=16)

    x_02 = Conv2D(num_classes, (3, 3), strides=(1, 1),
                  kernel_regularizer=regularization,
                  padding='valid')(x_2)
    x_02 = GlobalAveragePooling2D()(x_02)

    # module 3 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch3_11 = conv2d_bn(x_2, 64, 1, 1, strides=(1, 1),
                           kernel_regularizer=regularization, padding='same')
    branch3_12 = SeparableConv2D(128, (3, 3), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch3_11)
    branch3_12 = BatchNormalization()(branch3_12)
    branch3_12 = Activation('relu')(branch3_12)
    branch3_13 = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch3_12)
    branch3_13 = BatchNormalization()(branch3_13)
    branch3_13 = Activation('relu')(branch3_13)
    branch3_13 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_13)

    branch3_21 = Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2)
    branch3_21 = BatchNormalization()(branch3_21)
    branch3_21 = Activation('relu')(branch3_21)
    branch3_22 = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch3_21)
    branch3_22 = BatchNormalization()(branch3_22)
    branch3_22 = Activation('relu')(branch3_22)
    branch3_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_22)

    branch3_41 = SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(x_2)
    branch3_41 = BatchNormalization()(branch3_41)
    branch3_41 = Activation('relu')(branch3_41)

    x_3 = layers.concatenate([branch3_13, branch3_23, branch3_41],
                             name='mixed3')
    x_3 = squeeze_excite_block(x_3, ratio=16)

    x_03 = GlobalAveragePooling2D()(x_3)

    # module 4 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch4_11 = conv2d_bn(x_3, 128, 1, 1, strides=(1, 1),
                           kernel_regularizer=regularization, padding='same')
    branch4_12 = SeparableConv2D(256, (3, 3), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch4_11)
    branch4_12 = BatchNormalization()(branch4_12)
    branch4_12 = Activation('relu')(branch4_12)
    branch4_13 = SeparableConv2D(256, (3, 3), strides=(1, 1), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch4_12)
    branch4_13 = BatchNormalization()(branch4_13)
    branch4_13 = Activation('relu')(branch4_13)
    branch4_13 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_13)

    branch4_21 = Conv2D(128, (1, 1), strides=(1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3)
    branch4_21 = BatchNormalization()(branch4_21)
    branch4_21 = Activation('relu')(branch4_21)
    branch4_22 = SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(branch4_21)
    branch4_22 = BatchNormalization()(branch4_22)
    branch4_22 = Activation('relu')(branch4_22)
    branch4_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_22)

    branch4_41 = SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same',
                                 kernel_regularizer=regularization,
                                 use_bias=False)(x_3)
    branch4_41 = BatchNormalization()(branch4_41)
    branch4_41 = Activation('relu')(branch4_41)

    x_4 = layers.concatenate([branch4_13, branch4_22, branch4_41], name='mixed4')
    x_4 = squeeze_excite_block(x_4, ratio=16)

    x_04 = GlobalAveragePooling2D()(x_4)

    # output module
    x_mix = layers.concatenate([x_02, x_03, x_04], name='mixed_GAP')
    x = Dropout(rate=0.4)(x_mix)
    x = Dense(num_classes)(x)

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


# adding model concate
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

    x_1 = layers.concatenate([branch1_1, branch1_2, branch1_3],  name='mixed1')
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

    x_2 = layers.concatenate([branch2_1, branch2_2, branch2_3],  name='mixed2')
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

    x_3 = layers.concatenate([branch3_1, branch3_2, branch3_3],  name='mixed3')
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

    x_4 = layers.concatenate([branch4_1, branch4_2, branch4_3],  name='mixed4')
    x_4 = squeeze_excite_block(x_4, ratio=16)


    x_2 = GlobalAveragePooling2D()(x_2)
    x_3 = GlobalAveragePooling2D()(x_3)
    x_4 = GlobalAveragePooling2D()(x_4)

    # x_2 = GlobalMaxPooling2D()(x_2)
    # x_3 = GlobalMaxPooling2D()(x_3)
    # x_4 = GlobalMaxPooling2D()(x_4)

    x = layers.concatenate([x_2, x_3, x_4],  name='layer_concate')
    # x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax', name='predictions')(x)
    
    # instantiate a Model
    model = Model(img_input, output) 
    return model


def mini_concate_V4_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
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

    branch1_3 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x)
    branch1_3 = SeparableConv2D(16, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch1_3)
    branch1_3 = BatchNormalization()(branch1_3)

    branch1_4 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch1_4 = Activation('relu')(branch1_4)
    branch1_4 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch1_4)
    branch1_4 = BatchNormalization()(branch1_4)
    branch1_4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_4)

    x_1 = layers.concatenate(
        [branch1_1, branch1_2, branch1_3, branch1_4], name='mixed1')
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

    branch2_3 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_1)
    branch2_3 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch2_3)
    branch2_3 = BatchNormalization()(branch2_3)
    # branch2_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_3)

    branch2_4 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x_1)
    branch2_4 = BatchNormalization()(branch2_4)
    branch2_4 = Activation('relu')(branch2_4)
    branch2_4 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch2_4)
    branch2_4 = BatchNormalization()(branch2_4)
    branch2_4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_4)

    x_2 = layers.concatenate(
        [branch2_1, branch2_2, branch2_3, branch2_4], name='mixed2')
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

    branch3_3 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_2)
    branch3_3 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch3_3)
    branch3_3 = BatchNormalization()(branch3_3)

    branch3_4 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x_2)
    branch3_4 = BatchNormalization()(branch3_4)
    branch3_4 = Activation('relu')(branch3_4)
    branch3_4 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch3_4)
    branch3_4 = BatchNormalization()(branch3_4)
    branch3_4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_4)

    x_3 = layers.concatenate(
        [branch3_1, branch3_2, branch3_3, branch3_4], name='mixed3')
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

    branch4_3 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_3)
    branch4_3 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch4_3)
    branch4_3 = BatchNormalization()(branch4_3)

    branch4_4 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x_3)
    branch4_4 = BatchNormalization()(branch4_4)
    branch4_4 = Activation('relu')(branch4_4)
    branch4_4 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch4_4)
    branch4_4 = BatchNormalization()(branch4_4)
    branch4_4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_4)

    x_4 = layers.concatenate(
        [branch4_1, branch4_2, branch4_3, branch4_4], name='mixed4')
    x_4 = squeeze_excite_block(x_4, ratio=16)

    # output model
    x_2 = GlobalAveragePooling2D()(x_2)
    x_3 = GlobalAveragePooling2D()(x_3)
    x_4 = GlobalAveragePooling2D()(x_4)

    x = layers.concatenate([x_2, x_3, x_4], name='layer_concate')
    print x.shape
    shape = (1, 1, 896)
    x = Reshape(shape)(x)

    x = Dropout(0.5)(x)
    x = SeparableConv2D(num_classes, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Reshape((num_classes,))(x)
    output = Activation('softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output)
    return model





if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    # model = mini_XCEPTION(input_shape, num_classes)
    # model.summary()
    # model = Multi_XCEPTION(input_shape, num_classes)
    # model.summary()
    # model = SE_medium_XCEPTION(input_shape, num_classes)
    # model.summary()
    # model = SE_XCEPTION(input_shape, num_classes)   # parameters is 59095
    # model.summary()
    # model = Add_GAP_XCEPTION(input_shape, num_classes)   # parameters is 59095
    # model.summary()  
    # model = GAP_CNN_XCEPTION(input_shape, num_classes)   # parameters is 59095
    # model.summary()
    # model = Multi_bsb_drop_XCEPTION(input_shape, num_classes)   # parameters is 59095
    # model.summary()  
    # model = mini_concate_XCEPTION(input_shape, num_classes)
    # # model.summary()
    model = mini_concate_V3_XCEPTION(input_shape, num_classes)
    model.summary()
    model = plot_model(model, to_file='../../images/models/mini_concate_V3_0912_XCEPTION.png', show_shapes=True,
                       show_layer_names=True)

    # model = mini_FPN_Net(input_shape, num_classes)
    # model = mini_unsampling_Xception(input_shape, num_classes)
    # model.summary()
