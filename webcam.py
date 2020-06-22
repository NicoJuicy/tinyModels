import cv2, os, PIL
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img


cam = cv2.VideoCapture(0)
cv2.namedWindow("TinyFace Webcam Demo")
img_counter = 0

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale = 1
fontColor_green = (0,128,0)
fontColor_red = (0,0,128)
fontColor_yellow = (255,255,0)
lineType = 2

model_path = "tinyFace.tflite"
input_image_dim = (32, 32, 3)

def face_detector(image):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img1 = load_img(image, target_size=(input_image_dim[0], input_image_dim[1]), interpolation="nearest")
    input_data = np.empty((1, input_image_dim[0], input_image_dim[1], input_image_dim[2]), dtype=np.float32)
    img = img_to_array(img1)
    input_data[0, :, :, :] = img
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    processed_image = input_data[0,:,:,:]

    threshold_face = 0.7
    threshold_object = 0.5

    if output_data[0, 0] >= threshold_face and output_data[0, 1] <= threshold_object:
        print("Positive: ", output_data)
        cv2.putText(processed_image, "P", bottomLeftCornerOfText, font, fontScale, fontColor_green, lineType)
    elif output_data[0, 0] <= threshold_face and output_data[0, 1] >= threshold_object:
        print("Negative: ", output_data)
        cv2.putText(processed_image, "N", bottomLeftCornerOfText, font, fontScale, fontColor_red, lineType)
    else:
        print("Not sure: ", output_data)
        cv2.putText(processed_image, "?", bottomLeftCornerOfText, font, fontScale, fontColor_yellow, lineType)

    cv2.imwrite(f"{image}", processed_image)
    print(f"{img_name} written!")


while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("TinyFace Webcam Demo", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("ESC hit, closing")
        break
    elif k % 256 == 32:
        if not os.path.isdir("webcam_results"):
            os.mkdir("webcam_results")
        img_name = "webcam_results/frame_{}.png".format(img_counter)
        frame = frame[80:-80, 160:480, :]

        frame = cv2.resize(frame, (input_image_dim[0], input_image_dim[1]), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(img_name, frame)
        img_counter += 1
        face_detector(img_name)

cam.release()
cv2.destroyAllWindows()
