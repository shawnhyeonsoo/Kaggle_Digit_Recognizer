import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


f = pd.read_csv("train.csv")
train_images = [list(f.iloc[i][1:]) for i in range(len((f)))]
train_label = [f.iloc[i][0] for i in range(len(f))]
train_images = np.array(train_images).reshape(-1,28,28,1)
train_labels = np.array(train_label)

t = pd.read_csv("test.csv")
test_images = [list(t.iloc[i]) for i in range(len((t)))]
test_images = np.array(test_images).reshape(-1,28,28,1)

train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50)


results = []
for i in range(len(test_images)):
    results.append(int(tf.argmax(model.predict(test_images[i:i+1]),1)))

import csv
list1 = [1+i for i in range(len(test_images))]
list2 = results

d = zip(list1, list2)
with open('output_mnist_deep_cnn.csv', 'w',encoding = 'utf8') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("ImageId", "Label"))
    wr.writerows(d)
myfile.close()
