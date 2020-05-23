import sys, argparse, pathlib
from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD


def vgg_3():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(Conv2D( 64,(3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    # optimizer = "adam" woulda worked too
    return model
