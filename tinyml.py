import sys, argparse, pathlib
import numpy as np
from matplotlib import pyplot
from math import floor
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
# from models.vgg_3 import vgg_3
# from models.squeezenet import SqueezeNet
# from models.squeezenet_opt import squeezenet
from models.squeezenet_tiny import squeezenet


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
    default=256,
)

parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    help="Type in how many training epochs you want to have ",
    default=50,
)
parser.add_argument(
    "-m",
    "--model",
    help="Choose model to be used for training: [vgg_3][squeezenet_full][squeezenet_simplified][squeezenet_quantized]",
    default="vgg_3",
)
args = parser.parse_args()


def load_dataset(dataset_path="export/dataset.npy", labels_path="export/labels.npy", training_perc=0.8):
    trainX = np.empty((100000 + 11633 + 610126, 32, 32, 3))
    trainY = np.empty((100000 + 11633 + 610126, 1))
    testX = np.empty((20000 + 2000 + 11000, 32, 32, 3))
    testY = np.empty((20000 + 2000 + 11000, 1))
    (trainX_10, trainY_10), (testX_10, testY_10) = cifar10.load_data()
    (trainX_100, trainY_100), (testX_100, testY_100) = cifar100.load_data()
    trainY_10 = to_categorical(trainY_10)
    testY_10 = to_categorical(testY_10)
    trainY_100 = to_categorical(trainY_100)
    testY_100 = to_categorical(testY_100)
    dataset_lfwild = np.load("export/dataset_lfwild.npy")
    dataset_aligned_images = np.load("export/dataset_aligned_images.npy")
    trainX = np.row_stack((trainX_10, trainX_100, dataset_lfwild[0:11633], dataset_aligned_images[0:610126]))
    trainY = np.row_stack((np.zeros((50000, 1)), np.zeros((50000, 1)), np.ones((11633, 1)), np.ones((610126, 1))))
    testX = np.row_stack((testX_10, testX_100, dataset_lfwild[11633:], dataset_aligned_images[610126:]))
    testY = np.row_stack((np.zeros((10000, 1)), np.zeros((10000, 1)), np.ones((2000, 1)), np.ones((11000, 1))))
    # labels = np.load(labels_path)
    # dataset_size = np.shape(dataset)[0]
    # training_len = floor(dataset_size * training_perc)
    # trainX = dataset[0:training_len]
    # testX = dataset[training_len:dataset_size]
    # trainY = labels[0:training_len]
    # testY = labels[training_len:dataset_size]

    print(np.shape(trainX))
    print(np.shape(testX))
    print(np.shape(trainY))
    print(np.shape(testY))

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
    # model = squeezenet(classes=2) # face det
    model = squeezenet(classes=2) # for cifar 10
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    quantize_model = tfmot.quantization.keras.quantize_model
    quantized_model = quantize_model(model)
    quantized_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    save_keras_full = True
    if save_keras_full:
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