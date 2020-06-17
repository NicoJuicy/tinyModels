import os, PIL

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import load_img

orig_path = "dataset/lfwild"
orig_path_2 = "dataset/aligned_images"

saved = 0
saved_backgr = 0
width = 32
height = 32
negatives = 10000
size_lfwild = 13633
size_aligned_images = 621126
# pic_array = np.empty((size_lfwild + size_aligned_images + negatives, width, height, 3), dtype=np.uint8)
pic_array = np.empty((size_aligned_images, width, height, 3), dtype=np.uint8)
labels_array = np.empty((size_lfwild, 1))

# for pic in os.listdir("dataset/negatives"):
#     img_array = load_img(f"dataset/negatives/{pic}", target_size=(32, 32), interpolation="nearest")
#     pic_array[size_lfwild + size_aligned_images + saved_backgr, :, :, :] = img_to_array(img_array)
#     labels_array[size_lfwild + size_aligned_images + saved_backgr] = 0
#
#     saved_backgr += 1

# for folder in os.listdir(orig_path):
#     for pic in os.listdir(f"{orig_path}/{folder}"):
#         img_array = load_img(f"{orig_path}/{folder}/{pic}", target_size=(32,32), interpolation="nearest")
#         pic_array[saved, :, :, :] = img_to_array(img_array)
#         labels_array[saved] = 1
#
#
#         # img_array_full = PIL.Image.open(f"{orig_path}/{folder}/{pic}")
#         # img_array_background = img_to_array(img_array_full)
#         # img_array_background = img_array_background[-32:, :32, :]
#         #
#         # if saved_backgr < negatives:
#         #     pic_array[saved_backgr + size_lfwild + size_aligned_images, :, :, :] = img_to_array(img_array_background)
#         #     labels_array[size_lfwild + size_aligned_images + saved_backgr] = 0
#
#         saved += 1
#         # saved_backgr += 1

for folder in os.listdir(orig_path_2):
    for subfolder in os.listdir(f"{orig_path_2}/{folder}"):
        for pic in os.listdir(f"{orig_path_2}/{folder}/{subfolder}"):
            img_array = load_img(f"{orig_path_2}/{folder}/{subfolder}/{pic}", target_size=(32, 32), interpolation="nearest")
            pic_array[saved, :, :, :] = img_to_array(img_array)
            # labels_array[saved] = 1

            saved += 1



print(f"Saved {saved} image samples")

if not os.path.isdir("export"):
    os.mkdir("export")

np.save(f"export/dataset_aligned_images", pic_array)
# np.save(f"export/labels_lfwild", labels_array)

