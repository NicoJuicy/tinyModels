import numpy as np
import os, hashlib
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


dataset_path = "path"
export_path = "path"

for person in os.listdir(dataset_path):
    for image in os.listdir(f"{dataset_path}/{person}"):
        img = load_img(f"{dataset_path}/{person}/{image}")
        img_array = img_to_array(img)
        np_array = np.asarray(img_array)
        np.save(f"{export_path}/{hashlib.sha1(image).hexdigest()[:10]}.npy", np_array)

dataset_np = []
for face in os.listdir(export_path):
    dataset_np.append(face)
np.save("dataset.npy", dataset_np)

