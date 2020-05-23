import sys, argparse, pathlib
from matplotlib import pyplot
from numpy import shape
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from models.vgg_3 import vgg_3

parser = argparse.ArgumentParser(
    description="Automatic model optimizer"
)

parser.add_argument(
    "-v",
    "--version",
    action="version",
    version=f"Automatic NN model optimizer version 1.0",
)

parser.add_argument(
    "-b",
    "--batchsize",
    type=int,
    help="Type in how many samples you want in one training batch "
         "default is 64",
    default=64,
)

parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    help="Type in how many training epochs you want to have ",
    default=100,
)
parser.add_argument(
    "-m",
    "--model",
    help="Choose model to be used for training: [vgg_3]",
    default="vgg_3",
)
args = parser.parse_args()


def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    # print(shape(trainX), shape(trainY))
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    # converting from uint8 to float32
    train_norm = train.astype("float32")
    test_norm = test.astype("float32")
    # normalizing to range 0 to 1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


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
    model = vgg_3()
    history = model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(testX, testY),
        shuffle=True
    )
    results = model.evaluate(testX, testY)
    print("Loss, Accuracy:", results)
    summarize_diagnostics(history)

    # saving this stuff
    model_structure = model.to_json()
    f = pathlib.Path("model_structure.json")
    f.write_text(model_structure)

    model.save_weights("model_weights.h5")


# entry point
number_epochs = args.epochs
batch_len = args.batchsize
model_choice = args.model
run_training(epochs=number_epochs, batch_size=batch_len)


# # TODO: perform optimizations for TinyML
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