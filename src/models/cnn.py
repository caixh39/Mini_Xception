import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, UpSampling2D
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


def mini_FPN_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, padding='same',
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print 'input layers x:', x.shape

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

    # x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x_1 = layers.add([x, residual])
    print 'model_1 layers x_1:', x.shape

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

    # x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x_2 = layers.add([x, residual])
    print 'model_2 layers x_2:', x_2.shape

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

    # x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x_3 = layers.add([x, residual])
    print 'model_3 layers x_3:', x_3.shape

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

    # x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x_4 = layers.add([x, residual])
    print 'model_4 layers x_4:', x_4.shape
    
    # stage_3
    x_44 = UpSampling2D()(x_4)

    x_33 = SeparableConv2D(128, (1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3)
    x_11 = BatchNormalization()(x_33)

    x_3_add = layers.add([x_44, x_33], name='unsampling_1')
    x_3_add = SeparableConv2D(224, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3_add)
    x_3_add = BatchNormalization()(x_3_add)
    print 'upsampling layers x_3:', x_33.shape, x_3_add.shape

    # stage_2

    x_33 = UpSampling2D(size=(2, 2))(x_3_add)

    x_22 = SeparableConv2D(224, (1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2)
    x_11 = BatchNormalization()(x_22)

    x_2_add = layers.add([x_22, x_33], name='unsampling_2')
    x_2_add = SeparableConv2D(224, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2_add)
    x_2_add = BatchNormalization()(x_2_add)
    print 'upsampling layers x_2:', x_2.shape, x_2_add.shape

    # stage_1
    x_22 = UpSampling2D(size=(2, 2))(x_2_add)

    x_11 = SeparableConv2D(224, (1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_1)
    x_11 = BatchNormalization()(x_11)

    x_1_add = layers.add([x_11, x_22], name='unsampling_3')
    x_1_add = SeparableConv2D(224, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_1_add)
    x_1_add = BatchNormalization()(x_1_add)
    print 'upsampling layers x_1:', x_11.shape, x_1_add.shape

    # output model
    x_1_add = GlobalAveragePooling2D()(x_1_add)
    x_2_add = GlobalAveragePooling2D()(x_2_add)
    x_3_add = GlobalAveragePooling2D()(x_3_add)

    x = layers.concatenate([x_1_add, x_2_add, x_3_add], name='mixed_predict')
    x = Dense(num_classes)(x)
    output = Activation('softmax', name='predictions')(x)
    
    # instantiate a Model
    model = Model(img_input, output) 
    return model


def mini_FPN_Net(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base, getting multi scale info
    img_input = Input(input_shape)

    x = Conv2D(8, (1, 1), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)                                           
    x = BatchNormalization()(x)
    x = Conv2D(8, (1, 1), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    branch_0 = BatchNormalization()(x) 

    x = Conv2D(8, (1, 1), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = SeparableConv2D(16, (5, 5), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)                                           
    x = BatchNormalization()(x)
    x = Conv2D(8, (1, 1), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    branch_1 = BatchNormalization()(x)

    x_0 = layers.concatenate([branch_0, branch_1])       


    # module 1
    x = Conv2D(16, (1, 1), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x_0)
    x = BatchNormalization()(x)    
    x = SeparableConv2D(16, (3, 3), strides=(2, 2), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x_1 = squeeze_excite_block(x, ratio=16)


    # module 2
    x = Conv2D(32, (1, 1), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x_1)
    x = BatchNormalization()(x) 

    x = SeparableConv2D(32, (3, 3), padding='same', strides=(2, 2),
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x_2 = squeeze_excite_block(x, ratio=16)


    # module 3
    x = Conv2D(64, (1, 1), strides=(1, 1),
                      padding='same', use_bias=False)(x_2)
    x = BatchNormalization()(x)

    x = SeparableConv2D(64, (3, 3), padding='same', strides=(2, 2),
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x_3 = squeeze_excite_block(x, ratio=16)


    # module 4
    x = Conv2D(128, (1, 1), strides=(1, 1),
                      padding='same', use_bias=False)(x_3)
    x = BatchNormalization()(x)

    x = SeparableConv2D(128, (3, 3), padding='same', strides=(2, 2),
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x_4 = squeeze_excite_block(x, ratio=16)
    print x_4.shape


    # stage_3
    x_44 = UpSampling2D()(x_4)

    x_33 = SeparableConv2D(128, (1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3)
    x_11 = BatchNormalization()(x_33)

    x_3_add = layers.add([x_44, x_33], name='unsampling_1')
    x_3_add = SeparableConv2D(224, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_3_add)
    x_3_add = BatchNormalization()(x_3_add)
    print 'upsampling layers x_3:', x_33.shape, x_3_add.shape

    # stage_2

    x_33 = UpSampling2D(size=(2, 2))(x_3_add)

    x_22 = SeparableConv2D(224, (1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2)
    x_11 = BatchNormalization()(x_22)

    x_2_add = layers.add([x_22, x_33], name='unsampling_2')
    x_2_add = SeparableConv2D(224, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_2_add)
    x_2_add = BatchNormalization()(x_2_add)
    print 'upsampling layers x_2:', x_22.shape, x_2_add.shape

    # stage_1
    x_22 = UpSampling2D(size=(2, 2))(x_2_add)

    x_11 = SeparableConv2D(224, (1, 1), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_1)
    x_11 = BatchNormalization()(x_11)

    x_1_add = layers.add([x_11, x_22], name='unsampling_3')
    x_1_add = SeparableConv2D(224, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x_1_add)
    x_1_add = BatchNormalization()(x_1_add)
    print 'upsampling layers x_1:', x_11.shape, x_1_add.shape

    # output model
    x_1_add = GlobalAveragePooling2D()(x_1_add)
    x_2_add = GlobalAveragePooling2D()(x_2_add)
    x_3_add = GlobalAveragePooling2D()(x_3_add)

    x = layers.concatenate([x_1_add, x_2_add, x_3_add], name='mixed_predict')
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
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
    # model = mini_MSE_XCEPTION(input_shape, num_classes)
    # model.summary()
    model = mini_FPN_Net(input_shape, num_classes)
    model.summary()