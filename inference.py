import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# platform = "keras"
platform = "tflite"


# For running full-size model inference
if platform == "keras":
    model_path = "model_full.h5"
    model = tf.keras.models.load_model(model_path)

    # inference prototype: result = model.predict(image)

    dataset_file = "export/dataset.npy"

    positives = 0
    negatives = 0

    for one_image in os.listdir("dataset/test"):
        if (positives + negatives) < 50:

            img1 = load_img(f"dataset/test/{one_image}")
            img_array = np.empty((1, 32, 32, 3), dtype=np.uint8)
            img = img_to_array(img1)
            img_array[0, :, :, :] = img
            result = model.predict(img_array)

        if result[0, 0] == 0 and result[0, 1] == 1:
           positives += 1
        elif result[0, 0] == 1 and result[0, 1] == 0:
            negatives += 1

    print(f"Positives: {positives / (positives + negatives) * 100}%")
    print(f"Negatives: {negatives / (positives + negatives) * 100}%")


# For running TFlite reduced-size model inference
elif platform == "tflite":
    model_path = "tinyFace.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']

    positives = 0
    negatives = 0

    for one_image in os.listdir("dataset/test"):
        if (positives + negatives) < 200:

            img1 = load_img(f"dataset/test/{one_image}", target_size=(32,32))
            input_data = np.empty((1, 32, 32, 3), dtype=np.float32)
            img = img_to_array(img1)
            input_data[0, :, :, :] = img
            # result = model.predict(img_array)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(one_image, output_data)

        if output_data[0, 0] >= 0.9 and output_data[0, 1] <= 0.1:
           positives += 1
        elif output_data[0, 0] <= 0.1 and output_data[0, 1] >= 0.9:
            negatives += 1

    print(f"Positives: {positives / (positives + negatives) * 100}%")
    print(f"Negatives: {negatives / (positives + negatives) * 100}%")

