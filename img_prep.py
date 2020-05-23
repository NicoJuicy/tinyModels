import numpy as np
import os, hashlib
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


dataset_path = "dataset"
export_path = "export"

if not os.path.isdir(dataset_path):
    os.makedirs(dataset_path)
    print("You need to fill up the dataset folder first")
if not os.path.isdir(export_path):
    os.makedirs(export_path)

folder_size = len(os.listdir(dataset_path))
saved = 0
width = 32
height = 32
big_array = np.empty((folder_size,width,height,3), dtype=np.uint8)
print(big_array)
for image in os.listdir(dataset_path):
    img = load_img(f"{dataset_path}/{image}")
    big_array[saved,:,:,:] = img_to_array(img)
    saved += 1
    print(f"Processed {saved}/{folder_size}")

# np.save(f"{export_path}/{hashlib.sha1(image.encode()).hexdigest()}", big_array) # goofy name
np.save (f"{export_path}/dataset", big_array)

