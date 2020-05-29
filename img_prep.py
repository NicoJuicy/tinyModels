import numpy as np
import os, hashlib
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


dataset_path_faces = "dataset/faces"
dataset_path_nonfaces = "dataset/non_faces"
export_path = "export"

if not os.path.isdir(dataset_path_faces):
    os.makedirs(dataset_path_faces)
    print("You need to fill up the dataset folder first")
if not os.path.isdir(export_path):
    os.makedirs(export_path)

folder_size = len(os.listdir(dataset_path_faces)) + len(os.listdir(dataset_path_nonfaces))
saved = 0
width = 32
height = 32
big_array = np.empty((folder_size,width,height,3), dtype=np.uint8)
labels = np.empty((folder_size), dtype=np.uint8)
print(big_array)
# load faces
for image in os.listdir(dataset_path_faces):
    img = load_img(f"{dataset_path_faces}/{image}")
    big_array[saved,:,:,:] = img_to_array(img)
    labels[saved] = 1 # 1 means True, this is a positive face
    saved += 1
    print(f"Processed {saved}/{folder_size}")

# load random non-face objects
for image in os.listdir(dataset_path_nonfaces):
    img = load_img(f"{dataset_path_nonfaces}/{image}")
    big_array[saved,:,:,:] = img_to_array(img)
    labels[saved] = 0 # 0 means False, this is a negative, a non-face
    saved += 1
    print(f"Processed {saved}/{folder_size}")

# np.save(f"{export_path}/{hashlib.sha1(image.encode()).hexdigest()}", big_array) # goofy name
np.save(f"{export_path}/dataset", big_array)
np.save(f"{export_path}/labels", labels)


# TODO
# generate negatives in a smarter manner