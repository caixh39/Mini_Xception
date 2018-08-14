import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from keras import backend as K
from keras.layers import GlobalAveragePooling2D
from keras.layers import Reshape, Dense, multiply, Permute


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor'''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    # kernel_initializer='he_normal', use_bias=False
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_uniform', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_uniform', use_bias=False)(se)
    
    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x