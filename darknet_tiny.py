import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model
model = keras.Sequential()

model.add(keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=(224,224,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), input_shape=(224, 224, 16)))
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=(112,112,16)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), input_shape=(112, 112, 32)))

model.add(keras.layers.Conv2D(16, kernel_size=(1,1), input_shape=(56,56,32)))
model.add(keras.layers.Conv2D(128, kernel_size=(3,3), input_shape=(56,56,16)))
model.add(keras.layers.Conv2D(16, kernel_size=(1,1), input_shape=(56,56,128)))
model.add(keras.layers.Conv2D(128, kernel_size=(3,3), input_shape=(56,56,16)))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), input_shape=(56, 56, 128)))

model.add(keras.layers.Conv2D(32, kernel_size=(1,1), input_shape=(28,28,128)))
model.add(keras.layers.Conv2D(256, kernel_size=(3,3), input_shape=(28,28,32)))
model.add(keras.layers.Conv2D(32, kernel_size=(1,1), input_shape=(28,28,256)))
model.add(keras.layers.Conv2D(256, kernel_size=(3,3), input_shape=(28,28,32)))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), input_shape=(28,28,256)))

model.add(keras.layers.Conv2D(64, kernel_size=(1,1), input_shape=(14,14,256)))
model.add(keras.layers.Conv2D(512, kernel_size=(3,3), input_shape=(14,14,64)))
model.add(keras.layers.Conv2D(64, kernel_size=(1,1), input_shape=(14,14,512)))
model.add(keras.layers.Conv2D(512, kernel_size=(3,3), input_shape=(14,14,64)))
model.add(keras.layers.Conv2D(128, kernel_size=(1,1), input_shape=(14,14,512)))
model.add(keras.layers.Conv2D(1000, kernel_size=(1,1), input_shape=(14,14,128)))

model.add(keras.layers.AvgPool2D(input_shape=(14,14,1000)))
model.add(keras.layers.Softmax())

quantized_model = quantize_model(model)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

quantize_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


batch_size = 100
epochs = 100

print("Normal")
history = model.fit(x=images_train,y=labels_train, epochs=epochs, batch_size=batch_size)
model.summary()

print("Quantized model")
history = quantized_model.fit(x=images_train,y=labels_train, epochs=epochs, batch_size=batch_size)
model.summary()

res = model.evaluate(images_test, labels_test)
print("Model1 has an accuracy of {0:.2f}%".format(res[1] * 100))

res = quantized_model.evaluate(images_test, labels_test)
print("Model1 has an accuracy of {0:.2f}%".format(res[1] * 100))


converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

open("model.tflite", "wb").write(tflite_model)

# do xxd magic here, try this out in the terminal
# xxd -i model.tflite > model.cc

# TODO: perform optimizations for TinyML