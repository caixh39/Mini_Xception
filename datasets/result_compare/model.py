def mini_concate_V4_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(2, 2), kernel_regularizer=regularization,
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
    branch1_1 = Activation('relu')(branch1_1)

    branch1_2 = SeparableConv2D(16, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch1_2 = BatchNormalization()(branch1_2)
    branch1_2 = Activation('relu')(branch1_2)
    branch1_2 = SeparableConv2D(16, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch1_2)
    branch1_2 = BatchNormalization()(branch1_2)
    branch1_2 = Activation('relu')(branch1_2)

    branch1_3 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x)
    branch1_3 = SeparableConv2D(16, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch1_3)
    branch1_3 = BatchNormalization()(branch1_3)
    branch1_3 = Activation('relu')(branch1_3)

    branch1_4 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x)
    branch1_4 = BatchNormalization()(branch1_4)
    branch1_4 = Activation('relu')(branch1_4)
    branch1_4 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch1_4)
    branch1_4 = BatchNormalization()(branch1_4)
    branch1_4 = Activation('relu')(branch1_4)
    branch1_4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch1_4)

    x_1 = layers.concatenate(
        [branch1_1, branch1_2, branch1_3, branch1_4], name='mixed1')
    x_1 = squeeze_excite_block(x_1, ratio=16)

    # module 2
    branch2_1 = Conv2D(32, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x_1)
    branch2_1 = BatchNormalization()(branch2_1)
    branch2_1 = Activation('relu')(branch2_1)

    branch2_2 = SeparableConv2D(32, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x_1)
    branch2_2 = BatchNormalization()(branch2_2)
    branch2_2 = Activation('relu')(branch2_2)
    branch2_2 = SeparableConv2D(32, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch2_2)
    branch2_2 = BatchNormalization()(branch2_2)
    branch2_2 = Activation('relu')(branch2_2)

    branch2_3 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_1)
    branch2_3 = SeparableConv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch2_3)
    branch2_3 = BatchNormalization()(branch2_3)
    branch2_3 = Activation('relu')(branch2_3)
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
    branch2_4 = Activation('relu')(branch2_4)
    branch2_4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch2_4)

    x_2 = layers.concatenate(
        [branch2_1, branch2_2, branch2_3, branch2_4], name='mixed2')
    x_2 = squeeze_excite_block(x_2, ratio=16)

    # module 3
    branch3_1 = Conv2D(64, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x_2)
    branch3_1 = BatchNormalization()(branch3_1)
    branch3_1 = Activation('relu')(branch3_1)

    branch3_2 = SeparableConv2D(64, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x_2)
    branch3_2 = BatchNormalization()(branch3_2)
    branch3_2 = Activation('relu')(branch3_2)
    branch3_2 = SeparableConv2D(64, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch3_2)
    branch3_2 = BatchNormalization()(branch3_2)
    branch3_2 = Activation('relu')(branch3_2)

    branch3_3 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_2)
    branch3_3 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch3_3)
    branch3_3 = BatchNormalization()(branch3_3)
    branch3_3 = Activation('relu')(branch3_3)

    branch3_4 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x_2)
    branch3_4 = BatchNormalization()(branch3_4)
    branch3_4 = Activation('relu')(branch3_4)
    branch3_4 = SeparableConv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch3_4)
    branch3_4 = BatchNormalization()(branch3_4)
    branch3_4 = Activation('relu')(branch3_4)
    branch3_4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch3_4)

    x_3 = layers.concatenate(
        [branch3_1, branch3_2, branch3_3, branch3_4], name='mixed3')
    x_3 = squeeze_excite_block(x_3, ratio=16)

    # module 4
    branch4_1 = Conv2D(128, (1, 1), strides=(2, 2),
                       padding='same', use_bias=False)(x_3)
    branch4_1 = BatchNormalization()(branch4_1)
    branch4_1 = Activation('relu')(branch4_1)

    # branch4_2 = Activation('relu')(x_3)
    branch4_2 = SeparableConv2D(128, (1, 1), padding='same', strides=(1, 1),
                                kernel_regularizer=regularization,
                                use_bias=False)(x_3)
    branch4_2 = BatchNormalization()(branch4_2)
    branch4_2 = Activation('relu')(branch4_2)
    branch4_2 = SeparableConv2D(128, (3, 3), padding='same', strides=(2, 2),
                                kernel_regularizer=regularization,
                                use_bias=False)(branch4_2)
    branch4_2 = BatchNormalization()(branch4_2)
    branch4_2 = Activation('relu')(branch4_2)

    branch4_3 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_3)
    branch4_3 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch4_3)
    branch4_3 = BatchNormalization()(branch4_3)
    branch4_3 = Activation('relu')(branch4_3)

    branch4_4 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(x_3)
    branch4_4 = BatchNormalization()(branch4_4)
    branch4_4 = Activation('relu')(branch4_4)
    branch4_4 = SeparableConv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularization,
                                use_bias=False)(branch4_4)
    branch4_4 = BatchNormalization()(branch4_4)
    branch4_4 = Activation('relu')(branch4_4)
    branch4_4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(branch4_4)

    x_4 = layers.concatenate(
        [branch4_1, branch4_2, branch4_3, branch4_4], name='mixed4')
    x_4 = squeeze_excite_block(x_4, ratio=16)

    # output model
    x_2 = GlobalAveragePooling2D()(x_2)
    x_3 = GlobalAveragePooling2D()(x_3)
    x_4 = GlobalAveragePooling2D()(x_4)

    x = layers.concatenate([x_2, x_3, x_4], name='layer_concate')
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax', name='predictions')(x)

    # instantiate a Model
    model = Model(img_input, output)
    return model

