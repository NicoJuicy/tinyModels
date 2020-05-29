import tensorflow.keras as keras
from numpy import shape, load, save
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

model_path = "model_full.h5"
model = keras.models.load_model(model_path)

# inference prototype: result = model.predict(image)

dataset_file = "export/dataset.npy"
dataset = load(dataset_file)
result = model.predict(dataset)
# print(shape(dataset))
# print(result)
save(f"inference_res", result)