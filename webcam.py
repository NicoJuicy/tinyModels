import cv2, os, PIL
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img


cam = cv2.VideoCapture(0)
cv2.namedWindow("TFLite Inference")
img_counter = 0

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale = 1
fontColor = (255,255,255)
lineType = 2

def face_detector(image):
    model_path = "tinyFace.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # input_shape = input_details[0]["shape"]

    img1 = load_img(image, target_size=(32, 32), interpolation="nearest")
    input_data = np.empty((1, 32, 32, 3), dtype=np.float32)
    img = img_to_array(img1)
    input_data[0, :, :, :] = img
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    threshold = 0.9

    if output_data[0, 0] >= threshold:
        print("Positive: ", output_data)
        return "POSITIVE"
    elif output_data[0, 0] <= threshold:
        print("Negative: ", output_data)
        return "NEGATIVE"
    else:
        print("Not sure: ", output_data)
        return "NO IDEA"


while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("TFLite Inference", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        # frame = cv2.resize(frame, dsize=(32, 32), interpolation=cv2.INTER_AREA)
        # face_detector(frame)
        if not os.path.isdir("webcam_lib"):
            os.mkdir("webcam_lib")
        img_name = "webcam_lib/opencv_frame_{}.png".format(img_counter)
        # img_array_cropped = img_to_array(frame)
        frame = frame[80:-80, 160:480, :]
        # img_array_cropped = array_to_img(img_array_cropped)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        result = face_detector(img_name)
        cv2.putText(frame, result, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        print(result)
        # cv2.imwrite(img_name, frame)

cam.release()

cv2.destroyAllWindows()
