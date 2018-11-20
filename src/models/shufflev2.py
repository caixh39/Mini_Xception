#''' coding:utf-8 -*-
# Created on 18-8-14
#'''
import numpy as np
from keras.utils import plot_model
from keras.engine.topology import get_source_inputs
from keras.layers import Input, Conv2D, MaxPool2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dense
from keras.models import Model
import keras.backend as K
from utils import block

def ShuffleNetV2(input_shape, num_classes,
                 scale_factor=1.0,
                 load_model=None,
                 num_shuffle_units=[3,7,3],
                 bottleneck_ratio=1):
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)


    img_input = Input(shape=input_shape)

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    x = GlobalAveragePooling2D(name='global_avg_pool')(x)


    x = Dense(num_classes, name='fc')(x)
    x = Activation('softmax', name='softmax')(x)


    inputs = img_input

    model = Model(inputs, x, name=name)

    if load_model:
        model.load_weights('', by_name=True)

    return model

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    input_shape = [64,64,1]
    num_class = 7
    model = ShuffleNetV2(input_shape=input_shape, num_classes=num_class, bottleneck_ratio=1)
    plot_model(model, to_file='shufflenetv2.png', show_layer_names=True, show_shapes=True)


    pass
