import tensorflow.keras as keras
import tensorflow as tf
from numpy import shape, load, save
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

platform = "keras"
platform = "tflite"

# For running full-size model inference
if platform == "keras":
    model_path = "model_full.h5"
    model = keras.models.load_model(model_path)

    # inference prototype: result = model.predict(image)

    dataset_file = "export/dataset.npy"
    dataset = load(dataset_file)
    result = model.predict(dataset)
    # print(shape(dataset))
    print(result)

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
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)