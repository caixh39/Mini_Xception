# 3 branch: Conv2D + SeparableConv2D(1x1+3x3) + SeparableConv2D(3x3+3x3)
def mini_concateV2_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
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

    x = layers.concatenate([branch1_1, branch1_2, branch1_3],  name='mixed1')
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

    x = layers.concatenate([branch2_1, branch2_2, branch2_3],  name='mixed2')
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

    x = layers.concatenate([branch3_1, branch3_2, branch3_3],  name='mixed3')
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

    x = layers.concatenate([branch4_1, branch4_2, branch4_3],  name='mixed4')
    x = squeeze_excite_block(x, ratio=16)

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    
    # instantiate a Model
    model = Model(img_input, output) 
    return model




