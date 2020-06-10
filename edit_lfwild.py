import os, PIL

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import load_img

orig_path = "dataset/lfwild"

saved = 0
saved_backgr = 0
width = 32
height = 32
negatives = 6000
pic_array = np.empty((13633 + negatives, width, height, 3), dtype=np.uint8)
labels_array = np.empty((13633 + negatives, 1))

for pic in os.listdir("dataset/negatives"):
    img_array = load_img(f"dataset/negatives/{pic}", target_size=(32, 32), interpolation="nearest")
    pic_array[13633 + saved_backgr, :, :, :] = img_to_array(img_array)
    labels_array[13633 + saved_backgr] = 0

    saved_backgr += 1

for folder in os.listdir(orig_path):
    for pic in os.listdir(f"{orig_path}/{folder}"):
        img_array = load_img(f"{orig_path}/{folder}/{pic}", target_size=(32,32), interpolation="nearest")
        pic_array[saved, :, :, :] = img_to_array(img_array)
        labels_array[saved] = 1


        img_array_full = PIL.Image.open(f"{orig_path}/{folder}/{pic}")
        img_array_background = img_to_array(img_array_full)
        img_array_background = img_array_background[-32:, :32, :]

        if saved_backgr < negatives:
            pic_array[saved_backgr + 13633, :, :, :] = img_to_array(img_array_background)
            labels_array[13633 + saved_backgr] = 0

        saved += 1
        saved_backgr += 1


print(f"Saved {saved + saved_backgr} image samples")

if not os.path.isdir("export"):
    os.mkdir("export")

np.save(f"export/dataset", pic_array)
np.save(f"export/labels", labels_array)

