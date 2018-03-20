print('loading packages')
import numpy as np
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from matplotlib.pyplot import imshow
from PIL import Image
# from __future__ import print_function
from sklearn.utils import shuffle
import cv2
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection  import GridSearchCV

print('generatin aux functions')
def data_train_extraction(folder_train):
    labels = [x[0].split('/')[-1] for x in os.walk(folder_train)][1:]
    TRAIN_FOLDER_ALL = [folder_train + '/' + x for x in labels]

    full_labels = list()
    imagepaths = list()

    n = 0
    for label in labels:
        for x in os.walk(TRAIN_FOLDER_ALL[n]):
            for y in x[2]:
                imagepaths.append(TRAIN_FOLDER_ALL[n] + '/' + y)
                full_labels.append(n)

        n += 1

    return imagepaths, full_labels, labels


def data_test_extraction(folder_test):
    images_test = [x for x in os.walk(folder_test)][0][2]
    TEST_FOLDER_ALL = [folder_test + '/' + x for x in images_test]

    return TEST_FOLDER_ALL


def load_m(model_n):
    model_l = load_model('models/{}.h5'.format(model_n))
    model_l.load_weights('weights/{}.h5'.format(model_n))
    return model_l


def submit(model, data, sub_name):
    path_test = [df_test['path'][i].split('/')[-1] for i in range(len(df_test['path']))]
    pr1 = model.predict(data)
    pr2 = label_binarizer.inverse_transform(pr1)
    names_pr = [number_to_names[str(i)] for i in pr2]
    submiss = pd.DataFrame({'file': path_test, 'species': names_pr})
    submiss.to_csv('submissions/{}'.format(sub_name), index=None)


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def proc_image0(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def proc_image1(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def normalize_scale(image_data):
    a = -0.5
    b = 0.5
    scale_min = 0
    scale_max = 255
    return a + ( ( (image_data - scale_min)*(b - a) )/( scale_max - scale_min ) )

def image_edit0(path, resolution):
    img = cv2.imread(path)
    (b, g, r)=cv2.split(img)
    img=cv2.merge([r,g,b])
    image = misc.imresize(img, (resolution, resolution), mode=None)
    return image

print('reading df data')
df_train = pd.read_csv('data/df_data/train.csv')
df_test = pd.read_csv('data/df_data/test.csv')
# if not os.path.exists('data/df_data'):
#     os.makedirs('data/df_data')
#
# if not os.path.isfile('data/df_data/train.csv'):
#     train.to_csv('data/df_data/train.csv', index=False)
# else:
#     df_train = pd.read_csv('data/df_data/train.csv')
#
# if not os.path.isfile('data/df_data/test.csv'):
#     test.to_csv('data/df_data/test.csv', index=False)
# else:
#     df_test = pd.read_csv('data/df_data/test.csv')



number_to_names = {'0': 'Common wheat',
                  '1': 'Fat Hen',
                  '2': 'Small-flowered Cranesbill',
                  '3': 'Maize',
                  '4': 'Scentless Mayweed',
                   '5': 'Cleavers',
                  '6': 'Charlock',
                  '7': 'Sugar beet',
                  '8': 'Shepherds Purse',
                  '9': 'Black-grass',
                  '10': 'Common Chickweed',
                  '11': 'Loose Silky-bent'}


print('processing images to numpy')
# images_train = [image_edit0(path, 32) for path in df_train['path']]
# np.savez("images_train", images_train)
images_train = np.load("images_train.npz")
array_train = np.asarray(images_train['arr_0'])
# images_test = [image_edit0(path, 32) for path in df_test['path']]
# np.savez("images_test", images_test)
images_test = np.load("images_test.npz")
array_test = np.asarray(images_test['arr_0'])


print('train/test split and shuffle')
X_train, X_test, y_train, y_test =train_test_split(array_train,
                                                   df_train['label'],
                                                   test_size=0.2,
                                                   random_state=42,
                                                   stratify = df_train['label']
                                                   )

X_train, y_train = shuffle(X_train, y_train)

X_normalized = normalize_scale(X_train)
X_normalized_test = normalize_scale(X_test)
test_normalized = normalize_scale(array_test)

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
y_one_hot_test = label_binarizer.fit_transform(y_test)

print('Start model training and evaluation')


model = Sequential()

model.add(Convolution2D(16, 4, 4, input_shape=(32, 32, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.25))

model.add(Convolution2D(32, 4, 4, activation="relu"))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))

# model.add(Convolution2D(128, 4, 4, activation="relu"))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.25))

model.add(Dense(12, activation="softmax"))

BATCH_SIZE = 128
EPOCHS = 10
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
gen = ImageDataGenerator(
            rotation_range=360.,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0,
            horizontal_flip=True,
            vertical_flip=True
    )

earlyStopping=keras.callbacks.EarlyStopping(monitor='acc', patience=4, verbose=1, mode='auto')

model.fit_generator(gen.flow(X_normalized, y_one_hot,batch_size=BATCH_SIZE),
               steps_per_epoch=100,
               epochs=EPOCHS,
               verbose=1,
               shuffle=True,
                    callbacks=[earlyStopping],
               validation_data=(X_normalized_test, y_one_hot_test))


history = model.fit(X_normalized,
                    y_one_hot,
                    steps_per_epoch=20,
                    epochs=10,
                    verbose=1,
                    callbacks=[earlyStopping]
                   )

preds = model.predict(X_normalized_test)
predictions = label_binarizer.inverse_transform(preds)
f1_s = round(f1_score(y_test, predictions, average='micro'),4)

model_name = 'modelo_{}'.format(str(f1_s))
model.save_weights('weights/{}.h5'.format(model_name))
model.save('models/{}.h5'.format(model_name))

model_load = load_model('models/{}.h5'.format(model_name))
model_load.load_weights('weights/{}.h5'.format(model_name))
test_score = model_load.evaluate(X_normalized_test, y_one_hot_test)
print(test_score)

# submit(model_load, test_normalized, 'f1s_{}.csv'.format(str(f1_s)))