import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images_resized = np.array([tf.image.resize(image, (32, 32)) for image in train_images])
train_images_resized = np.repeat(train_images_resized, 3, axis=-1)

test_images_resized = np.array([tf.image.resize(image, (32, 32)) for image in test_images])
test_images_resized = np.repeat(test_images_resized, 3, axis=-1)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images_resized, train_labels, epochs=10, batch_size=64, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images_resized, test_labels)
print('Test accuracy:', test_acc)
