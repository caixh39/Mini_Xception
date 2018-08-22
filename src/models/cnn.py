import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers import Reshape, Dense, multiply, Permute
from keras import layers
from keras.regularizers import l2
from keras import backend as K
from Module_Net import squeeze_excite_block, conv2d_bn


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
    
    x = Activation('relu')(x)
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

    x = Activation('relu')(x)
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

    x = Activation('relu')(x)
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
    x = Activation('relu')(x)
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

def mini_MSE_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = Activation('relu')(x)
    # x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
    #            use_bias=False)(x)
    # x = Activation('relu')(x)

    branch0_1 = conv2d_bn(x, 8, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='valid')
    branch0_1 = SeparableConv2D(16, (3, 3), padding='valid',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch0_1)                   
    branch0_1 = BatchNormalization()(branch0_1)
    branch0_1 = Activation('relu')(branch0_1)
    branch0_1 = SeparableConv2D(16, (3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch0_1)
    branch0_1 = BatchNormalization()(branch0_1)
    # branch0_1 = Activation('relu')(branch0_1)

    
    branch0_2 = conv2d_bn(x, 8, 1, 1, strides=(1, 1), padding='valid')
    branch0_2 = SeparableConv2D(16, (3, 3), strides=(1, 1), padding='valid',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch0_2)
    branch0_2 = BatchNormalization()(branch0_2)



    branch0_3 = SeparableConv2D(8, (3, 3), strides=(1, 1), padding='valid',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    branch0_3 = BatchNormalization()(branch0_3)
    branch0_3 = Activation('relu')(branch0_3)
    # branch1_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_12)

    x_0 = layers.concatenate([branch0_1, branch0_2, branch0_3], name='mixed0')
    x_0 = squeeze_excite_block(x_0, ratio=16)

    # x_0 = Conv2D(num_classes, (3, 3), strides=(1, 1), padding='valid',
    #             kernel_regularizer=regularization)(x_0)
    x_00 = GlobalAveragePooling2D()(x_0)



    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x_0)
    residual = BatchNormalization()(residual)

    
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_0)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x) 
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = squeeze_excite_block(x, ratio=16)
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
    x = squeeze_excite_block(x, ratio=16)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = squeeze_excite_block(x, ratio=16)
    x = layers.add([x, residual])

    # module 
    x = Conv2D(64, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)

    x_1 = GlobalAveragePooling2D()(x)
    x_1 = layers.concatenate([x_1, x_00], name='mixed_GAP')
    print x_1.shape

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

    x = squeeze_excite_block(x, ratio=16)
    x_3 = layers.add([x, residual])


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

    x = squeeze_excite_block(x, ratio=16)
    x = layers.add([x, residual])

    
    # x_2 = layers.concatenate([x_2, x_1], name='mixed_GAP')

    # module 
    x = Conv2D(num_classes, (3, 3), strides=(1, 1), 
                kernel_regularizer=regularization,
                use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    output = Activation('softmax', name='predictions')(x)
    
    # instantiate a Model
    model = Model(img_input, output) 
    return model

def SE_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
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

    # module 0
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x_1 = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x_1)

    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = squeeze_excite_block(x, ratio=16)
    x = layers.add([x, x_1])
    
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x_1 = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x_1)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = squeeze_excite_block(x, ratio=16)
    x = layers.add([x, x_1])

    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Squeeze and Excitation  module



    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    # residual = Activation('relu')(residual)

    x_2 = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x_2)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = squeeze_excite_block(x, ratio=16)
    x = layers.add([x, x_2])

    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Squeeze and Excitation  module
    # x = squeeze_excite_block(x, ratio=16)

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x_3 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x_3)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=16)
    x = layers.add([x, x_3])
    
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Squeeze and Excitation  module
    # x = squeeze_excite_block(x, ratio=16)

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x_4 = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x_4)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = squeeze_excite_block(x, ratio=16)
    x = layers.add([x, x_4])

    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # output moduel
    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    
    # instantiate a Model
    model = Model(img_input, output) 
    return model


def Add_GAP_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
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
    x_0 = Activation('relu')(x)

    x_01 = Conv2D(num_classes, (3, 3), strides=(1, 1),
               # kernel_regularizer=regularization,
               padding='same')(x_0)
    x_01 = GlobalAveragePooling2D()(x_01)
    

    # module 1 branch1
    branch1_11 =  SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_0)

    branch1_12 = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(branch1_11)                     
    branch1_12 = BatchNormalization()(branch1_12)
    branch1_12 = Activation('relu')(branch1_12)
    branch1_13 = squeeze_excite_block(branch1_12, ratio=16)
    
    branch1_21 = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_0)
    branch1_21 = BatchNormalization()(branch1_21)
    branch1_21 = Activation('relu')(branch1_21)
    branch1_22 = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_21)
    branch1_22 = BatchNormalization()(branch1_22)

    branch1_22 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_22)

    x_1 = layers.concatenate([branch1_13, branch1_22],
                            name='mixed1')


    x_01 = Conv2D(num_classes, (3, 3), strides=(1, 1),
               # kernel_regularizer=regularization,
               padding='same')(x_1)
    x_01 = GlobalAveragePooling2D()(x_01)


    # module 2 branch1
    branch2_11 =  SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_1)

    branch2_12 = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(branch2_11)                     
    branch2_12 = BatchNormalization()(branch2_12)
    branch2_12 = Activation('relu')(branch2_12)
    branch2_13 = squeeze_excite_block(branch2_12, ratio=16)
    
    branch2_21 = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_1)
    branch2_21 = BatchNormalization()(branch2_21)
    branch2_21 = Activation('relu')(branch2_21)

    branch2_22 = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_21)
    branch2_22 = BatchNormalization()(branch2_22)

    branch2_22 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_22)

    x_2 = layers.concatenate([branch2_13, branch2_22],
                            name='mixed2')


    x_02 = Conv2D(num_classes, (3, 3), strides=(1, 1),
               # kernel_regularizer=regularization,
               padding='same')(x_2)
    x_02 = GlobalAveragePooling2D()(x_02)


    # module 3 branch1
    branch3_11 =  SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2)

    branch3_12 = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(branch3_11)                     
    branch3_12 = BatchNormalization()(branch3_12)
    branch3_12 = Activation('relu')(branch3_12)
    branch3_13 = squeeze_excite_block(branch3_12, ratio=16)
    
    branch3_21 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2)
    branch3_21 = BatchNormalization()(branch3_21)
    branch3_21 = Activation('relu')(branch3_21)

    branch3_22 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch3_21)
    branch3_22 = BatchNormalization()(branch3_22)
    branch3_22 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_22)

    x_3 = layers.concatenate([branch3_13, branch3_22],
                            name='mixed3')


    x_03 = Conv2D(num_classes, (3, 3), strides=(1, 1),
               # kernel_regularizer=regularization,
               padding='same')(x_3)
    x_03 = GlobalAveragePooling2D()(x_03)


    # module 4 branch1
    branch4_11 =  SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3)

    branch4_12 = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(branch4_11)                     
    branch4_12 = BatchNormalization()(branch4_12)
    branch4_12 = Activation('relu')(branch4_12)
    branch4_13 = squeeze_excite_block(branch4_12, ratio=16)

    
    branch4_21 = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3)
    branch4_21 = BatchNormalization()(branch4_21)
    branch4_21 = Activation('relu')(branch4_21)

    branch4_22 = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch4_21)
    branch4_22 = BatchNormalization()(branch4_22)

    branch4_22 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_22)

    # x_1 = layers.concatenate([branch1_13, branch1_22],
    x_4 = layers.add([branch4_13, branch4_22])                        # name='mixed0')


    x_04 = Conv2D(num_classes, (3, 3), strides=(1, 1),
               # kernel_regularizer=regularization,
               padding='same')(x_4)
    x_04 = GlobalAveragePooling2D()(x_04)


    # output module
    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x_4)
    x = GlobalAveragePooling2D()(x)

    x = layers.add([x, x_03, x_04])
    output = Activation('softmax', name='predictions')(x)
    
    # instantiate a Model
    model = Model(img_input, output) 
    return model

# accuracy is 66.75%, parameters is 290k or 322k
def Multi_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = conv2d_bn(img_input, 8, 3, 3, strides=(1, 1), 
                    kernel_regularizer=regularization, padding='valid')
    x_0 = conv2d_bn(x, 8, 3, 3, strides=(1, 1), 
                    kernel_regularizer=regularization, padding='valid')

    x_00 = Conv2D(num_classes, (3, 3), strides=(1, 1), padding='valid',
                kernel_regularizer=regularization)(x_0)
    x_00 = GlobalAveragePooling2D()(x_00)
    

    # module 1 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch1_11 = conv2d_bn(x_0, 16, 1, 1, strides=(2, 2), 
                            kernel_regularizer=regularization, padding='same')
    branch1_12 = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_11)                   
    branch1_12 = BatchNormalization()(branch1_12)
    branch1_12 = Activation('relu')(branch1_12)
    branch1_13 =  SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_12)
    branch1_13 = BatchNormalization()(branch1_13)
    branch1_13 = Activation('relu')(branch1_13)

    
    branch1_21 = conv2d_bn(x_0, 16, 1, 1, strides=(2, 2), padding='same')
    branch1_22 = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_21)
    branch1_22 = BatchNormalization()(branch1_22)
    branch1_22 = Activation('relu')(branch1_22)


    branch1_31 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_0)
    branch1_32 = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization,
                                use_bias=False)(branch1_31)
    branch1_32 = BatchNormalization()(branch1_32)
    branch1_32 = Activation('relu')(branch1_32)

    branch1_41 = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x_0)
    branch1_41 = BatchNormalization()(branch1_41)

    x_1 = layers.concatenate([branch1_13, branch1_22, branch1_32,branch1_41], name='mixed1')
    X_1 = squeeze_excite_block(x_1, ratio=16)

    x_01 = Conv2D(num_classes, (3, 3), strides=(1, 1), padding='valid',
                kernel_regularizer=regularization)(x_1)
    x_01 = GlobalAveragePooling2D()(x_01)



  # module 2 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch2_11 = conv2d_bn(x_1, 32, 1, 1, strides=(2, 2), 
                            kernel_regularizer=regularization, padding='same')
    branch2_12 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_11)                   
    branch2_12 = BatchNormalization()(branch2_12)
    branch2_12 = Activation('relu')(branch2_12)
    branch2_13 =  SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_12)
    branch2_13 = BatchNormalization()(branch2_13)
    branch2_13 = Activation('relu')(branch2_13)

    
    branch2_21 = conv2d_bn(x_1, 16, 1, 1, strides=(2, 2), padding='same')
    branch2_22 = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_21)
    branch2_22 = BatchNormalization()(branch2_22)
    branch2_22 = Activation('relu')(branch2_22)


    branch2_31 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_1)
    branch2_32 = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization,
                                use_bias=False)(branch2_31)
    branch2_32 = BatchNormalization()(branch2_32)
    branch2_32 = Activation('relu')(branch2_32)

    branch2_41 = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x_1)
    branch2_41 = BatchNormalization()(branch2_41)

    x_2 = layers.concatenate([branch2_13, branch2_22, branch2_32, branch2_41],
                            name='mixed2')
    X_2 = squeeze_excite_block(x_2, ratio=16)


    x_02 = Conv2D(num_classes, (3, 3), strides=(1, 1),
                kernel_regularizer=regularization,
                padding='valid')(x_2)
    x_02 = GlobalAveragePooling2D()(x_02)


    # module 3 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch3_11 = conv2d_bn(x_2, 64, 1, 1, strides=(2, 2), 
                            kernel_regularizer=regularization, padding='same')
    branch3_12 = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch3_11)                   
    branch3_12 = BatchNormalization()(branch3_12)
    branch3_12 = Activation('relu')(branch3_12)
    branch3_13 =  SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch3_12)
    branch3_13 = BatchNormalization()(branch3_13)
    branch3_13 = Activation('relu')(branch3_13)

    
    branch3_21 = Conv2D(32, (1, 1), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2)
    branch3_21 = BatchNormalization()(branch3_21)
    branch3_21 = Activation('relu')(branch3_21)
    branch3_22 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch3_21)
    branch3_22 = BatchNormalization()(branch3_22)
    branch3_22 = Activation('relu')(branch3_22)

    branch3_31 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_2)
    branch3_32 = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization,
                                use_bias=False)(branch3_31)
    branch3_32 = BatchNormalization()(branch3_32)
    branch3_32 = Activation('relu')(branch3_32)

    branch3_41 = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x_2)
    branch3_41 = BatchNormalization()(branch3_41)

    x_3 = layers.concatenate([branch3_13, branch3_22, branch3_32, branch3_41],
                            name='mixed3')
    X_3 = squeeze_excite_block(x_3, ratio=16)


    x_03 = Conv2D(num_classes, (3, 3), strides=(1, 1),
                kernel_regularizer=regularization,
                padding='valid')(x_3)
    x_03 = GlobalAveragePooling2D()(x_03)


    # module 4 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch4_11 = conv2d_bn(x_3, 64, 1, 1, strides=(2, 2), 
                            kernel_regularizer=regularization, padding='same')
    branch4_12 = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch4_11)                   
    branch4_12 = BatchNormalization()(branch4_12)
    branch4_12 = Activation('relu')(branch4_12)
    branch4_13 =  SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch4_12)
    branch4_13 = BatchNormalization()(branch4_13)
    branch4_13 = Activation('relu')(branch4_13)

    
    branch4_21 = Conv2D(64, (1, 1), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3)
    branch4_21 = BatchNormalization()(branch4_21)
    branch4_21 = Activation('relu')(branch4_21)
    branch4_22 = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch4_21)
    branch4_22 = BatchNormalization()(branch4_22)
    branch4_22 = Activation('relu')(branch4_22)


    branch4_31 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_3)
    branch4_32 = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization,
                                use_bias=False)(branch4_31)
    branch4_32 = BatchNormalization()(branch4_32)
    branch4_32 = Activation('relu')(branch4_32)

    branch4_41 = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x_3)
    branch4_41 = BatchNormalization()(branch4_41)

    x_4 = layers.concatenate([branch4_13, branch4_22, branch4_32, branch4_41], name='mixed4')
    x_4 = squeeze_excite_block(x_4, ratio=16)


    # output module
    x = Conv2D(num_classes, (3, 3),
               kernel_regularizer=regularization,
               padding='same')(x_4)

    x = GlobalAveragePooling2D()(x)
    x = layers.add([x, x_00, x_01, x_02, x_03])
    output = Activation('softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output) 
    return model


# parameters is 162k, accuracy is 
def GAP_concate_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = conv2d_bn(img_input, 8, 3, 3, strides=(1, 1), 
                        kernel_regularizer=regularization, padding='valid')
    x = conv2d_bn(x, 8, 3, 3, strides=(2, 2), 
                            kernel_regularizer=regularization, padding='valid')

    branch0_1 = conv2d_bn(x, 8, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='valid')
    branch0_1 = SeparableConv2D(16, (3, 3), padding='valid',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch0_1)                   
    branch0_1 = BatchNormalization()(branch0_1)
    branch0_1 = Activation('relu')(branch0_1)
    branch0_1 = SeparableConv2D(16, (3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch0_1)
    branch0_1 = BatchNormalization()(branch0_1)
    branch0_1 = Activation('relu')(branch0_1)
    branch0_1 = squeeze_excite_block(branch0_1, ratio=16)
    
    branch0_2 = conv2d_bn(x, 8, 1, 1, strides=(1, 1), padding='valid')
    branch0_2 = SeparableConv2D(16, (3, 3), strides=(1, 1), padding='valid',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch0_2)
    branch0_2 = BatchNormalization()(branch0_2)
    branch0_2 = Activation('relu')(branch0_2)
    branch0_2 = squeeze_excite_block(branch0_2, ratio=16)


    branch0_3 = SeparableConv2D(8, (3, 3), strides=(1, 1), padding='valid',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    branch0_3 = BatchNormalization()(branch0_3)
    branch0_3 = Activation('relu')(branch0_3)
    # branch1_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_12)

    x_0 = layers.concatenate([branch0_1, branch0_2, branch0_3], name='mixed0')
    x_0 = squeeze_excite_block(x_0, ratio=16)

    # x_0 = Conv2D(num_classes, (3, 3), strides=(1, 1), padding='valid',
    #             kernel_regularizer=regularization)(x_0)
    x_00 = GlobalAveragePooling2D()(x_0)


    # module 1 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch1_1 = conv2d_bn(x_0, 16, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='same')
    branch1_1 = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_1)                   
    branch1_1 = BatchNormalization()(branch1_1)
    branch1_1 = Activation('relu')(branch1_1)
    branch1_1 = SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_1)
    branch1_1 = BatchNormalization()(branch1_1)
    branch1_1 = Activation('relu')(branch1_1)
    branch1_1 = conv2d_bn(branch1_1, 32, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='same')
    # branch1_13 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_13)
    branch1_1 = squeeze_excite_block(branch1_1, ratio=16)
    
    branch1_2 = conv2d_bn(x_0, 16, 1, 1, strides=(1, 1), padding='same')
    branch1_2 = SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_2)
    branch1_2 = BatchNormalization()(branch1_2)
    branch1_2 = Activation('relu')(branch1_2)
    # branch1_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_22)
    branch1_2 = squeeze_excite_block(branch1_2, ratio=16)


    branch1_3 = SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_0)
    branch1_3 = BatchNormalization()(branch1_3)
    branch1_3 = Activation('relu')(branch1_3)
    branch1_3 = squeeze_excite_block(branch1_3, ratio=16)
    # branch1_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_12)

    x_1 = layers.concatenate([branch1_1, branch1_2, branch1_3], name='mixed1')
    x_1 = squeeze_excite_block(x_1, ratio=16)

    # x_01 = Conv2D(num_classes, (3, 3), strides=(1, 1), padding='valid',
    #             kernel_regularizer=regularization)(x_1)
    x_01 = GlobalAveragePooling2D()(x_1)



    # module 2 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch2_1 = conv2d_bn(x_1, 32, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='same')
    branch2_1 = SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_1)                   
    branch2_1 = BatchNormalization()(branch2_1)
    branch2_1 = Activation('relu')(branch2_1)
    branch2_1 = SeparableConv2D(64, (3, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_1)
    branch2_1 = BatchNormalization()(branch2_1)
    branch2_1 = Activation('relu')(branch2_1)
    branch2_1 = conv2d_bn(branch2_1, 64, 1, 1, strides=(1, 1), 
                        kernel_regularizer=regularization, padding='same')
    branch2_1 = squeeze_excite_block(branch2_1, ratio=16)

    
    branch2_2 = conv2d_bn(x_1, 32, 1, 1, strides=(1, 1), padding='same')
    branch2_2 = SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_2)
    branch2_2 = BatchNormalization()(branch2_2)
    branch2_2 = Activation('relu')(branch2_2)
    # branch2_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_2)
    branch2_2 = squeeze_excite_block(branch2_2, ratio=16)


    branch2_3 = SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_1)
    branch2_3 = BatchNormalization()(branch2_3)
    branch2_3 = Activation('relu')(branch2_3)
    branch2_3 = squeeze_excite_block(branch2_3, ratio=16)

    x_2 = layers.concatenate([branch2_1, branch2_2,  branch2_3],
                            name='mixed2')
    x_2 = squeeze_excite_block(x_2, ratio=16)


    # x_02 = Conv2D(num_classes, (3, 3), strides=(1, 1),
    #             kernel_regularizer=regularization,
    #             padding='valid')(x_2)
    x_02 = GlobalAveragePooling2D()(x_2)


    # module 3 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch3_1 = conv2d_bn(x_2, 32, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='same')
    branch3_1 = SeparableConv2D(64, (3, 3), padding='same', strides=(1, 1),
                        kernel_regularizer=regularization,
                        use_bias=False)(branch3_1)                   
    branch3_1 = BatchNormalization()(branch3_1)
    branch3_1 = Activation('relu')(branch3_1)
    branch3_1 = SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch3_1)
    branch3_1 = BatchNormalization()(branch3_1)
    branch3_1 = Activation('relu')(branch3_1)
    branch3_1 = conv2d_bn(branch3_1, 128, 1, 1, strides=(1, 1), 
                        kernel_regularizer=regularization, padding='same')
    # branch3_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_13)
    branch3_1 = squeeze_excite_block(branch3_1, ratio=16)

    

    branch3_2 = conv2d_bn(x_2, 64, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='same')
    branch3_2 = SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch3_2)
    branch3_2 = BatchNormalization()(branch3_2)
    branch3_2 = Activation('relu')(branch3_2)
    # branch3_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_22)
    branch3_2 = squeeze_excite_block(branch3_2, ratio=16)


    branch3_3 = SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2)
    branch3_3 = BatchNormalization()(branch3_3)
    branch3_3 = Activation('relu')(branch3_3)
    branch3_3 = squeeze_excite_block(branch3_3, ratio=16)

    x_3 = layers.concatenate([branch3_1, branch3_2,  branch3_3],
                            name='mixed3')
    x_3 = squeeze_excite_block(x_3, ratio=16)


    # x_03 = Conv2D(num_classes, (3, 3), strides=(1, 1),
    #             kernel_regularizer=regularization,
    #             padding='valid')(x_3)
    x_03 = GlobalAveragePooling2D()(x_3)


    # module 4 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    branch4_1 = conv2d_bn(x_3, 32, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='same')
    branch4_1 = SeparableConv2D(64, (3, 3), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                        use_bias=False)(branch4_1)                   
    branch4_1 = BatchNormalization()(branch4_1)
    branch4_1 = Activation('relu')(branch4_1)
    branch4_1 = SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch4_1)
    branch4_1 = BatchNormalization()(branch4_1)
    branch4_1 = Activation('relu')(branch4_1)
    # branch4_13 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_13)
    branch4_1 = conv2d_bn(branch4_1, 256, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='same')
    branch4_1 = squeeze_excite_block(branch4_1, ratio=16)

    
    branch4_2 = conv2d_bn(x_3, 128, 1, 1, strides=(1, 1), 
                            kernel_regularizer=regularization, padding='same')
    branch4_2 = SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch4_2)
    branch4_2 = BatchNormalization()(branch4_2)
    branch4_2 = Activation('relu')(branch4_2)
    # branch4_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_2)
    branch4_2 = squeeze_excite_block(branch4_2, ratio=16)


    branch4_3 = SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3)
    branch4_3 = BatchNormalization()(branch4_3)
    branch4_3 = Activation('relu')(branch4_3)
    branch4_2 = squeeze_excite_block(branch4_3, ratio=16)

    x_4 = layers.concatenate([branch4_1, branch4_2, branch4_3], name='mixed4')
    x_4 = squeeze_excite_block(x_4, ratio=16)


    x_4 = Conv2D(256, (1, 1),
               kernel_regularizer=regularization,
               padding='same')(x_4)
    x_04 = GlobalAveragePooling2D()(x_4)

    # output module
    # x = GlobalAveragePooling2D()(x)
    x_mix = layers.concatenate([x_02, x_03, x_04], name='mixed_GAP')

    x = Dropout(rate=0.4)(x_mix)
    x = Dense(num_classes)(x)

    output = Activation('softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output) 
    return model


# refine parameters
def Multi_bsb_drop_XCEPTION(input_shape, num_classes, l2_regularization=0.01, rate_dropout=0.2):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = conv2d_bn(img_input, 8, 3, 3, strides=(1, 1), 
                    kernel_regularizer=regularization, padding='valid')
    x_0 = conv2d_bn(x, 8, 3, 3, strides=(1, 1), 
                    kernel_regularizer=regularization, padding='valid')

    # x_00 = Conv2D(num_classes, (5, 5), strides=(1, 1), padding='valid',
    #             kernel_regularizer=regularization)(x_0)
    # x_00 = Dropout(rate=rate_dropout)(x_00)
    # x_00 = GlobalAveragePooling2D()(x_00)
    

    # module 1 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    # branch1_11 = conv2d_bn(x_0, 16, 1, 1, strides=(2, 2), 
    #                         kernel_regularizer=regularization, padding='same')
    branch1_11 = SeparableConv2D(16, (1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_0)                   
    branch1_12 = BatchNormalization()(branch1_11)
    branch1_12 = Activation('relu')(branch1_12)

    branch1_13 =  SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_12)
    branch1_13 = BatchNormalization()(branch1_13)
    # branch1_13 = Activation('relu')(branch1_13)

    branch1_13 =  SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_13)
    branch1_13 = BatchNormalization()(branch1_13)
    branch1_13 = Activation('relu')(branch1_13)
    branch1_13 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_13)
    
    branch1_21 = conv2d_bn(x_0, 16, 1, 1, strides=(1, 1), padding='same')
    branch1_22 = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch1_21) 
    branch1_22 = BatchNormalization()(branch1_22)
    branch1_22 = Activation('relu')(branch1_22)
    branch1_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_22)


    # branch1_31 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_0)
    # branch1_32 = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization,
    #                             use_bias=False)(branch1_31)
    # branch1_32 = BatchNormalization()(branch1_32)
    # branch1_32 = Activation('relu')(branch1_32)

    # branch1_41 = Conv2D(32, (1, 1), strides=(2, 2),
    #                   padding='same', use_bias=False)(x_0)
    # branch1_41 = BatchNormalization()(branch1_41)

    x_1 = layers.concatenate([branch1_13, branch1_23], name='mixed1')
    x_1 = squeeze_excite_block(x_1, ratio=16)

    x_01 = Conv2D(num_classes, (3, 3), strides=(1, 1), padding='valid',
                kernel_regularizer=regularization)(x_1)
    x_01 = Dropout(rate=rate_dropout)(x_01)
    x_01 = GlobalAveragePooling2D()(x_01)



  # module 2 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    # branch2_11 = conv2d_bn(x_1, 8, 1, 1, strides=(2, 2), 
    #                         kernel_regularizer=regularization, padding='same')
    branch2_11 = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_1)                   
    branch2_12 = BatchNormalization()(x_1)
    branch2_12 = Activation('relu')(branch2_12)
    branch2_13 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_12)
    branch2_13 = BatchNormalization()(branch2_13)
    branch2_13 = Activation('relu')(branch2_13)
    branch2_14 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_13)                   
    branch2_14 = BatchNormalization()(branch2_14)
    branch2_14 = Activation('relu')(branch2_14)
    branch2_14 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_14)

    
    branch2_21 = conv2d_bn(x_1, 32, 1, 1, strides=(1, 1), padding='same')
    branch2_22 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch2_21)
    branch2_22 = BatchNormalization()(branch2_22)
    branch2_22 = Activation('relu')(branch2_22)
    branch2_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_22)


    # branch2_31 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_1)
    # branch2_32 = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization,
    #                             use_bias=False)(branch2_31)
    # branch2_32 = BatchNormalization()(branch2_32)
    # branch2_32 = Activation('relu')(branch2_32)

    # branch2_41 = Conv2D(8, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x_1)
    # branch2_41 = BatchNormalization()(branch2_41)

    x_2 = layers.concatenate([branch2_14, branch2_23],
                            name='mixed2')
    x_2 = squeeze_excite_block(x_2, ratio=16)


    x_02 = Conv2D(num_classes, (3, 3), strides=(1, 1),
                kernel_regularizer=regularization,
                padding='valid')(x_2)
    x_02 = Dropout(rate=rate_dropout)(x_02)
    x_02 = GlobalAveragePooling2D()(x_02)


    # # module 3 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    # branch3_11 = conv2d_bn(x_2, 16, 1, 1, strides=(2, 2), 
    #                         kernel_regularizer=regularization, padding='same')
    branch3_12 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2)                   
    branch3_12 = BatchNormalization()(branch3_12)
    branch3_12 = Activation('relu')(branch3_12)
    branch3_13 =  SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch3_12)
    branch3_13 = BatchNormalization()(branch3_13)
    branch3_13 = Activation('relu')(branch3_13)
    branch3_13 =  SeparableConv2D(128, (3, 3), padding='same',
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
    branch3_22 = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch3_21)
    branch3_22 = BatchNormalization()(branch3_22)
    branch3_22 = Activation('relu')(branch3_22)
    branch3_23 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_22)

    # branch3_31 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_2)
    # branch3_32 = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization,
    #                             use_bias=False)(branch3_31)
    # branch3_32 = BatchNormalization()(branch3_32)
    # branch3_32 = Activation('relu')(branch3_32)

    # branch3_41 = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x_2)
    # branch3_41 = BatchNormalization()(branch3_41)

    x_3 = layers.concatenate([branch3_13, branch3_23],
                            name='mixed3')
    x_3 = squeeze_excite_block(x_3, ratio=16)


    x_03 = Conv2D(num_classes, (3, 3), strides=(1, 1),
                kernel_regularizer=regularization,
                padding='valid')(x_3)
    x_03 = Dropout(rate=rate_dropout)(x_03)
    x_03 = GlobalAveragePooling2D()(x_03)


    # module 4 branch1~4:(1x1+3x3+3x3)+(1x1+3x3)+(Avg+3x3)+(1x1)
    # branch4_11 = conv2d_bn(x_3, 32, 1, 1, strides=(2, 2), 
    #                         kernel_regularizer=regularization, padding='same')
    branch4_11 = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3)                   
    branch4_12 = BatchNormalization()(branch4_11)
    branch4_12 = Activation('relu')(branch4_12)
    branch4_13 =  SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch4_12)
    branch4_13 = BatchNormalization()(branch4_13)
    branch4_13 = Activation('relu')(branch4_13)
    branch4_13 =  SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch4_13)
    branch4_13 = BatchNormalization()(branch4_13)
    branch4_13 = Activation('relu')(branch4_13)
    branch4_13 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_13)

    
    branch4_21 = Conv2D(32, (1, 1), strides=(1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3)
    branch4_21 = BatchNormalization()(branch4_21)
    branch4_21 = Activation('relu')(branch4_21)
    branch4_22 = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(branch4_21)
    branch4_22 = BatchNormalization()(branch4_22)
    branch4_22 = Activation('relu')(branch4_22)
    branch4_23 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(branch4_13)


    # branch4_31 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_3)
    # branch4_32 = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization,
    #                             use_bias=False)(branch4_31)
    # branch4_32 = BatchNormalization()(branch4_32)
    # branch4_32 = Activation('relu')(branch4_32)

    # branch4_41 = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x_3)
    # branch4_41 = BatchNormalization()(branch4_41)

    x_4 = layers.concatenate([branch4_13, branch4_23], name='mixed4')
    x_4 = squeeze_excite_block(x_4, ratio=16)


    # output module
    x = Conv2D(num_classes, (3, 3),
               kernel_regularizer=regularization,
               padding='same')(x_4)
    x = Dropout(rate=rate_dropout)(x)

    x = GlobalAveragePooling2D()(x)
    x = layers.add([x, x_02, x_03])
    output = Activation('softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output) 
    return model, conv2d_7


if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    # model = mini_XCEPTION(input_shape, num_classes)
    # model.summary()
    # model = SE_mini_XCEPTION(input_shape, num_classes)
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
    # model = GAP_concate_XCEPTION(input_shape, num_classes)
    # model.summary()
    # mini_MSE_XCEPTION
    model = mini_MSE_XCEPTION(input_shape, num_classes)
    model.summary()
