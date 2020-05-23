import sys
from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD


def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    # converting from uint8 to float32
    train_norm = train.astype("float32")
    test_norm = test.astype("float32")
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def define_model():
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
    return model


def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history["accuracy"], color="blue", label="train")
    pyplot.plot(history.history["val_accuracy"], color="orange", label="test")
    # save plot to file
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()


def run_training(epochs, batch_size):
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    model = define_model()
    history = model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(testX, testY),
    )
    results = model.evaluate(testX, testY)
    print("Loss, Accuracy:", results)
    summarize_diagnostics(history)


# entry point
run_training(epochs=5, batch_size=64)

# print("Quantized model")
# history = quantized_model.fit(x=images_train,y=labels_train, epochs=epochs, batch_size=batch_size)
# model.summary()
#
# res = model.evaluate(images_test, labels_test)
# print("Model1 has an accuracy of {0:.2f}%".format(res[1] * 100))
#
# res = quantized_model.evaluate(images_test, labels_test)
# print("Model1 has an accuracy of {0:.2f}%".format(res[1] * 100))
#
#
# converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# tflite_model = converter.convert()
#
# open("model.tflite", "wb").write(tflite_model)
#
# # do xxd magic here, try this out in the terminal
# # xxd -i model.tflite > model.cc
#
# # TODO: perform optimizations for TinyML
