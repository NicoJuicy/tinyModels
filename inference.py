import tensorflow as tf
import numpy as np

platform = "keras"
# platform = "tflite"

# For running full-size model inference
if platform == "keras":
    model_path = "model_full.h5"
    model = tf.keras.models.load_model(model_path)

    # inference prototype: result = model.predict(image)

    dataset_file = "export/dataset.npy"
    dataset = np.load(dataset_file)
    result = model.predict(dataset)
    # print(np.shape(dataset))
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
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_dataset = np.load("export/dataset.npy")
    input_data = np.array(input_dataset, dtype=np.float32)
    # interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)