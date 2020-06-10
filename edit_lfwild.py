import os, PIL

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import load_img

orig_path = "dataset/lfwild"

saved = 0
width = 32
height = 32
pic_array = np.empty((13233 * 2, width, height, 3), dtype=np.uint8)
labels_array = np.empty((13233 * 2, 1))

for folder in os.listdir(orig_path):
    for pic in os.listdir(f"{orig_path}/{folder}"):
        img_array = load_img(f"{orig_path}/{folder}/{pic}", target_size=(32,32), interpolation="nearest")
        pic_array[saved, :, :, :] = img_to_array(img_array)


        img_array_full = PIL.Image.open(f"{orig_path}/{folder}/{pic}")
        img_array_background = img_to_array(img_array_full)
        img_array_background = img_array_background[-32:, :32, :]

        print(np.shape(img_array_background))
        pic_array[saved + 13233, :, :, :] = img_to_array(img_array_background)

        saved += 1

for i in range(13233):
    labels_array[i, :] = 1
    labels_array[i + 13233, :] = 0

print(f"Saved {saved} image samples")

if not os.path.isdir("export"):
    os.mkdir("export")

np.save(f"export/dataset", pic_array)
np.save(f"export/labels", labels_array)

