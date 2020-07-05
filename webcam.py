import cv2, os, PIL
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
dimension = (64, 64)
grayscale = True


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

model_path = "squeezenet_opt.tflite"
input_image_dim = (dimension[0], dimension[1], 1 if grayscale else 0)
print(f"Working with images in {input_image_dim}")

def face_detector(image):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # img1 = load_img(image, target_size=(input_image_dim[0], input_image_dim[1]), interpolation="nearest")

    img1 = cv2.imread(image, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(image)
    img1 = cv2.resize(img1, (input_image_dim[0], input_image_dim[1]))
    img1 = img_to_array(img1)

    input_data = np.empty((1, input_image_dim[0], input_image_dim[1], input_image_dim[2]), dtype=np.float32)
    img = img_to_array(img1)
    input_data[0, :, :, :] = img
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    processed_image = input_data[0,:,:,:]

    print("          OBJECT | FACE")
    print(f"{output_data[0, 0] * 100} | {output_data[0, 1] * 100}")
    if output_data[0, 0] * 100  < 90 and output_data[0, 1] * 100 > 10:
        print("POSITIVE")
    else:
        print("NEGATIVE")

    # if output_data[0, 0] >= 0.01 and output_data[0, 1] >= 0.01:
    #     if output_data[0, 0] >= 0.9 and output_data[0, 0] <= 0.98 and output_data[0, 1] >= 0.15:
    #         print("POSITIVE: ", output_data)
    #         cv2.putText(processed_image, "P", bottomLeftCornerOfText, font, fontScale, fontColor_green, lineType)
    #     else:
    #         print("NEGATIVE: ", output_data)
    #         cv2.putText(processed_image, "N", bottomLeftCornerOfText, font, fontScale, fontColor_red, lineType)
    # else:
    #     print("NEGATIVE: ", output_data)
    #     cv2.putText(processed_image, "?", bottomLeftCornerOfText, font, fontScale, fontColor_yellow, lineType)

    cv2.imwrite(f"{image}", processed_image)
    # print(f"{img_name} written!")


signal = False

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("TinyFace Webcam Demo", frame)

    if not signal:
        start = time.perf_counter()
        signal = True
        timer = False
    if time.perf_counter() - start > 0.5:
        timer = True
        signal = False

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("ESC hit, closing")
        break
    # elif k % 256 == 32:
    elif timer:
        if not os.path.isdir("webcam_results"):
            os.mkdir("webcam_results")
        img_name = "webcam_results/frame_{}.png".format(img_counter)
        frame = frame[80:-80, 160:480, :]

        frame = cv2.resize(frame, (input_image_dim[0], input_image_dim[1]), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.normalize(frame, None, alpha=0, beta=10, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(img_name, frame)
        img_counter += 1
        face_detector(img_name)

cam.release()
cv2.destroyAllWindows()
