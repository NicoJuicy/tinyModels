from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate

# thanks to https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/squeezenet_architecture.py

def fire_mod(x, fire_id, squeeze=16, expand=64):
    squeeze1x1 = 'squeeze1x1'
    expand1x1 = 'expand1x1'
    expand3x3 = 'expand3x3'
    relu = 'relu.'
    fid = 'fire' + str(fire_id) + '/'

    # define the squeeze layer ~ (1,1) filter
    x = Conv2D(squeeze, (1, 1), padding='valid', name=fid + squeeze1x1)(x)
    x = Activation('relu', name=fid + relu + squeeze1x1)(x)

    # define the expand layer's (1,1) filters
    expand_1x1 = Conv2D(expand, (1, 1), padding='valid', name=fid + expand1x1)(x)
    expand_1x1 = Activation('relu', name=fid + relu + expand1x1)(expand_1x1)

    # define the expand layer's (3,3) filters
    expand_3x3 = Conv2D(expand, (3, 3), padding='same', name=fid + expand3x3)(x)
    expand_3x3 = Activation('relu', name=fid + relu + expand3x3)(expand_3x3)

    # Concatenate
    x = concatenate([expand_1x1, expand_3x3], axis=3, name=fid + 'concat')

    return x

def squeezenet(input_shape=(32, 32, 3), classes=10):
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_mod(x, fire_id=2, squeeze=16, expand=64)
    x = fire_mod(x, fire_id=3, squeeze=16, expand=64)

    x = fire_mod(x, fire_id=4, squeeze=32, expand=128)
    x = fire_mod(x, fire_id=5, squeeze=32, expand=128)
    x = Dropout(0.5, name='drop9')(x)

    x = Conv2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(img_input, out, name='squeezenet')

    return model