from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate

def fire_mod(x, fire_id, squeeze=16, expand=64):
    # initalize naming convention of components of the fire module
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
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

# def squeezenet():
#     model = Sequential()
#     model.add(Conv2D(96, (7,7), activation="relu", kernel_initializer="glorot_uniform",strides=(2, 2), padding="same", input_shape=(32, 32, 3)))
#     maxpool1 = (MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_first"))
#     fire2_squeeze = (Conv2D(16, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(maxpool1)
#     fire2_expand1 = (Conv2D(64, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire2_squeeze)
#     fire2_expand2 = (Conv2D(64, (3,3), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire2_squeeze)
#     model.add(Concatenate(axis=1)([fire2_expand1, fire2_expand2]))
#
#     fire3_squeeze = (Conv2D(16, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))
#     fire3_expand1 = (Conv2D(64, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire3_squeeze)
#     fire3_expand2 = (Conv2D(64, (3,3), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire3_squeeze)
#     model.add(Concatenate(axis=1)([fire3_expand1, fire3_expand2]))
#
#     fire4_squeeze = (Conv2D(32, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))
#     fire4_expand1 = (Conv2D(128, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire4_squeeze)
#     fire4_expand2 = (Conv2D(128, (3,3), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire4_squeeze)
#     model.add(Concatenate(axis=1)([fire4_expand1, fire4_expand2]))
#
#     maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_first")
#
#     fire5_squeeze = (Conv2D(32, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(maxpool4)
#     fire5_expand1 = (Conv2D(128, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire5_squeeze)
#     fire5_expand2 = (Conv2D(128, (3,3), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire5_squeeze)
#     model.add(Concatenate(axis=1)([fire5_expand1, fire5_expand2]))
#
#     fire6_squeeze = (Conv2D(48, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))
#     fire6_expand1 = (Conv2D(192, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire6_squeeze)
#     fire6_expand2 = (Conv2D(192, (3,3), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire6_squeeze)
#     model.add(Concatenate(axis=1)([fire6_expand1, fire6_expand2]))
#
#     fire7_squeeze = (Conv2D(48, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))
#     fire7_expand1 = (Conv2D(192, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire7_squeeze)
#     fire7_expand2 = (Conv2D(192, (3,3), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire7_squeeze)
#     model.add(Concatenate(axis=1)([fire7_expand1, fire7_expand2]))
#
#     fire8_squeeze = (Conv2D(64, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))
#     fire8_expand1 = (Conv2D(256, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire8_squeeze)
#     fire8_expand2 = (Conv2D(256, (3,3), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire8_squeeze)
#     model.add(Concatenate(axis=1)([fire8_expand1, fire8_expand2]))
#
#     maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_first")
#
#     fire9_squeeze = (Conv2D(64, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(maxpool8)
#     fire9_expand1 = (Conv2D(256, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire9_squeeze)
#     fire9_expand2 = (Conv2D(256, (3,3), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))(fire9_squeeze)
#     model.add(Concatenate(axis=1)([fire9_expand1, fire9_expand2]))
#
#     model.add(Dropout(0.5))
#     model.add(Conv2D(10, (1, 1), activation="relu", kernel_initializer="glorot_uniform", padding="valid", data_format="channels_first",)
#     model.add(GlobalAveragePooling2D(data_format="channels_first"))
#     model.add(Activation("softmax"))
#
#     model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#     return model
    