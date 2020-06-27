import sys, argparse, pathlib, random, cv2, os
import numpy as np
from keras_preprocessing.image import img_to_array, ImageDataGenerator
from matplotlib import pyplot
import tensorflow_model_optimization as tfmot
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from models.vgg_3 import vgg_3
from models.squeezenet import SqueezeNet
from models.squeezenet_opt import squeezenet
from imutils import paths
from models.lenet import LeNet


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
    default=512,
)

parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    help="Type in how many training epochs you want to have ",
    default=1,
)
parser.add_argument(
    "-m",
    "--model",
    help="Choose model to be used for training: [vgg_3][squeezenet_full][squeezenet_simplified][squeezenet_quantized]",
    default="vgg_3",
)

parser.add_argument(
    "-d",
    "--dataset",
    help="Path of dataset to be used for training",
    default="dataset/teo_generated",
)

parser.add_argument(
    "-g",
    "--grayscale",
    action="store_false"
)

parser.add_argument(
    "-w",
    "--width",
    type=int,
    help="Width of images to work with",
    default=100
)

parser.add_argument(
    "-ht",
    "--height",
    type=int,
    help="Height of images to work with",
    default=100
)

args = parser.parse_args()
train_keras = False
grayscale = args.grayscale
dimension = (args.width, args.height)
print(f"Will work with images of size {dimension} in grayscale-{grayscale}")

def load_dataset():
    data = []
    labels = []

    print("Loading dataset...")

    image_paths = sorted(list(paths.list_images(args.dataset)))
    random.shuffle(image_paths)
    dataset_size = len(image_paths)
    count = 0
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(image_path)
        image = cv2.resize(image, dimension)
        image = img_to_array(image)
        data.append(image)
        label = image_path.split(os.path.sep)[1]
        if label == "positives":
            label = 1
        elif label == "negatives":
            label = 0
        else:
            print("dubious label")
            raise Exception

        labels.append(label)
        count += 1
        print(f"Loaded {count}/{dataset_size} images", end="\r")

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # trainY = to_categorical(trainY, num_classes=2)
    # testY = to_categorical(testY, num_classes=2)

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    print("Loaded dataset with following dimensions: ")
    print(f"trainX: {np.shape(trainX)}")
    print(f"testX: {np.shape(testX)}")
    print(f"trainY: {np.shape(trainY)}")
    print(f"testY: {np.shape(testY)}")

    return trainX, trainY, testX, testY, aug


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
    trainX, trainY, testX, testY, aug = load_dataset()
    model = LeNet.build(classes=2, width=dimension[0], height=dimension[1], depth=1 if grayscale == True else 3)
    if train_keras == True:
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    quantize_model = tfmot.quantization.keras.quantize_model
    quantized_model = quantize_model(model)
    quantized_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print(f"Compiled model")

    if train_keras == True:
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

    print("Starting training on quantized model")

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