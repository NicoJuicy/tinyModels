from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Activation, Concatenate, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D

def squeezenet():
    model = Sequential()
    model.add(Conv2D(96, (7,7), activation="relu", kernel_initializer="glorot_uniform",strides=(2, 2), padding="same", input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format="channels_first"))
    model.add(Conv2D(16, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))
    model.add(Conv2D(64, (1,1), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first"))
    model.add(Conv2D(64, (3,3), activation="relu", kernel_initializer="glorot_uniform", padding="same", data_format="channels_first",))
    