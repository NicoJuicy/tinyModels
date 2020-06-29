import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
dimension = (32, 32)
grayscale = True

# platform = "keras"
platform = "tflite"
threshold = 0.9

positives = 0
negatives = 0
neutrals = 0

# For running full-size model inference
if platform == "keras":
    model_path = "model_full.h5"
    model = tf.keras.models.load_model(model_path)

    dataset_file = "export/dataset.npy"

    for one_image in os.listdir("dataset/test"):
        if (positives + negatives) < 50:
            img1 = load_img(f"dataset/test/{one_image}")
            img_array = np.empty((1, dimension[0], dimension[1], 1 if grayscale else 3), dtype=np.uint8)
            img = img_to_array(img1)
            img_array[0, :, :, :] = img
            result = model.predict(img_array)

        if result[0, 0] >= threshold:
            positives += 1
        elif result[0, 0] <= threshold:
            negatives += 1
        else:
            neutrals += 1

    print(f"Positives: {positives / (positives + negatives + neutrals) * 100}%")
    print(f"Negatives: {negatives / (positives + negatives + neutrals) * 100}%")


# For running TFlite reduced-size model inference
elif platform == "tflite":
    model_path = "squeezenet_opt.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]

    for one_image in os.listdir("dataset/test"):
        if (positives + negatives) < 200:

            img1 = cv2.imread(f"dataset/test/{one_image}", cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(f"dataset/test/{one_image}")
            img1 = cv2.resize(img1, dimension)
            input_data = np.empty((1, dimension[0], dimension[1], 1 if grayscale else 3), dtype=np.float32)
            img = img_to_array(img1)
            input_data[0, :, :, :] = img
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])

            if output_data[0, 0] >= threshold:
                positives += 1
                print("Positive: ", one_image, output_data)
            elif output_data[0, 0] <= threshold:
                negatives += 1
                print("Negative: ", one_image, output_data)
            else:
                neutrals += 1
                print("Not sure: ", one_image, output_data)
    #
    # print(f"Positives: {positives / (positives + negatives + neutrals) * 100}%")
    # print(f"Negatives: {negatives / (positives + negatives + neutrals) * 100}%")
