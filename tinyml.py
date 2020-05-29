import sys, argparse, pathlib
import numpy as np
from matplotlib import pyplot
from math import floor
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from models.vgg_3 import vgg_3
from models.squeezenet import SqueezeNet
from models.squeezenet_opt import squeezenet


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
    default=20,
)
parser.add_argument(
    "-m",
    "--model",
    help="Choose model to be used for training: [vgg_3][squeezenet_full][squeezenet_simplified][squeezenet_quantized]",
    default="vgg_3",
)
args = parser.parse_args()


def load_dataset(dataset_path="export/dataset.npy", labels_path="export/labels.npy", training_perc=1):
    # (trainX, trainY), (testX, testY) = cifar10.load_data()
    dataset = np.load(dataset_path)
    labels = np.load(labels_path)
    dataset_size = np.shape(dataset)[0]
    training_len = floor(dataset_size * training_perc)
    trainX = dataset[0:training_len, :, :, :]
    testX = dataset[0:training_len, :, :, :]
    trainY = (labels[0:training_len])
    testY = (labels[0:training_len])

    # trainY = to_categorical(trainY)
    # testY = to_categorical(testY)
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
    # pyplot.savefig(filename + "_plot.png")
    pyplot.show()


def run_training(epochs, batch_size):
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    # model = vgg_3()
    # model = SqueezeNet(nb_classes=10, inputs=(32, 32, 3))
    model = squeezenet(classes=2)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    quantize_model = tfmot.quantization.keras.quantize_model
    quantized_model = quantize_model(model)
    quantized_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(testX, testY),
        shuffle=True
    )

    model.summary()
    results = model.evaluate(testX, testY)
    print("Loss, Accuracy:", results)
    summarize_diagnostics(history)

    # saving this stuff
    model_structure = model.to_json()
    f = pathlib.Path("model_structure.json")
    f.write_text(model_structure)
    model.save_weights("model_weights.h5")
    model.save("model_full.h5")

    history_q = quantized_model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(testX, testY),
        shuffle=True
    )

    quantized_model.summary()
    results = quantized_model.evaluate(testX, testY)
    print("Loss, Accuracy:", results)
    summarize_diagnostics(history_q)

    # saving this other stuff
    q_model_structure = quantized_model.to_json()
    f = pathlib.Path("q_model_structure.json")
    f.write_text(q_model_structure)
    model.save_weights("q_model_weights.h5")

    # def representative_dataset_gen():
    #     for image in images_test:
    #         array = np.array(image)
    #         array = np.expand_dims(array, axis=0)
    #         yield ([array])

    converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] # experiment with this
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # quantization
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    # converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()

    open("tinyFace.tflite", "wb").write(tflite_model)
    # !xxd - i MNIST_full_quanitization.tflite > MNIST_full_quanitization.cc

    ## TODO: Pruning





# entry point
number_epochs = args.epochs
batch_len = args.batchsize
model_choice = args.model
run_training(epochs=number_epochs, batch_size=batch_len)