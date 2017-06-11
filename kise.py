

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Convolution2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
image_list = []
label_list = []

image_list_test = []
label_list_test = []
datagen = ImageDataGenerator(rotation_range=180,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,fill_mode='nearest')
for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue
    dir1 = "data/train/" + dir
    label = 0
    if dir == "maru":
        label = 0
    elif dir == "shikaku":
        label = 1
    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((100, 100)))
            print(filepath)
#            image = image.transpose(2, 0, 1)
            print(image.shape)
            image_list.append(image / 255.)
image_list = np.array(image_list)
Y = to_categorical(label_list)

for dir in os.listdir("data/test"):
    if dir == ".DS_Store":
        continue
    dir1 = "data/test/" + dir
    label = 0
    if dir == "maru":
        label = 0
    elif dir == "shikaku":
        label = 1
    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list_test.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((100, 100)))
            print(filepath)
#            image = image.transpose(2, 0, 1)
            print(image.shape)
            image_list_test.append(image / 255.)
image_list_test = np.array(image_list_test)
Y_test = to_categorical(label_list_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(100,100,3)))
model.add(Activation("relu"))
model.add(Convolution2D(32, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode=("same")))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))


opt = Adam(lr=0.0001)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
datagen.fit(image_list)
model.fit_generator(datagen.flow(image_list, Y,batch_size=50),samples_per_epoch=1000, nb_epoch=300,validation_data=(image_list_test, Y_test))

