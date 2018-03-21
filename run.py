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


class Model(object):

    def __init__(self):
        if not os.path.exists('models'):
            os.makedirs('models')

        if not os.path.exists('submissions'):
            os.makedirs('submissions')

        if not os.path.exists('weights'):
            os.makedirs('weights')

        self.number_to_names = {'0': 'Common wheat',
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

        self.label_binarizer = LabelBinarizer()

    def data_train_extraction(self, folder_train):
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

    def data_test_extraction(self, folder_test):
        images_test = [x for x in os.walk(folder_test)][0][2]
        TEST_FOLDER_ALL = [folder_test + '/' + x for x in images_test]

        return TEST_FOLDER_ALL

    def load_m(self, model_n):
        model_l = load_model('models/{}.h5'.format(model_n))
        model_l.load_weights('weights/{}.h5'.format(model_n))
        return model_l

    def submit(self, model, data, sub_name, df_test):
        path_test = [df_test['path'][i].split('/')[-1] for i in range(len(df_test['path']))]
        pr1 = model.predict(data)
        pr2 = self.label_binarizer.inverse_transform(pr1)
        names_pr = [self.number_to_names[str(i)] for i in pr2]
        submiss = pd.DataFrame({'file': path_test, 'species': names_pr})
        submiss.to_csv('submissions/{}'.format(sub_name), index=None)

    def create_mask_for_plant(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        sensitivity = 35
        lower_hsv = np.array([60 - sensitivity, 100, 50])
        upper_hsv = np.array([60 + sensitivity, 255, 255])

        mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def proc_image0(self, image):
        mask = self.create_mask_for_plant(image)
        output = cv2.bitwise_and(image, image, mask=mask)
        return output

    def proc_image1(self, image):
        image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
        image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
        return image_sharp

    def normalize_scale(self, image_data):
        a = -0.5
        b = 0.5
        scale_min = 0
        scale_max = 255
        return a + (((image_data - scale_min) * (b - a)) / (scale_max - scale_min))

    def image_edit0(self, path, resolution):
        img = cv2.imread(path)
        (b, g, r) = cv2.split(img)
        img = cv2.merge([r, g, b])
        image = misc.imresize(img, (resolution, resolution), mode=None)
        return image

    def proc_or_load(self):
        if not os.path.exists('data/df_data'):
            os.makedirs('data/df_data')

        if not os.path.isfile('data/df_data/train.csv'):
            imagepaths, full_labels, labels = self.data_train_extraction('data/train')
            train = {'path': imagepaths, 'label': full_labels}
            train = pd.DataFrame(train)
            train.to_csv('data/df_data/train.csv', index=False)
            df_train = train
        else:
            df_train = pd.read_csv('data/df_data/train.csv')

        if not os.path.isfile('data/df_data/test.csv'):
            TEST_FOLDER_ALL = self.data_test_extraction('data/test')
            test = {'path': TEST_FOLDER_ALL}
            test = pd.DataFrame(test)
            test.to_csv('data/df_data/test.csv', index=False)
            df_test = test
        else:
            df_test = pd.read_csv('data/df_data/test.csv')

        return df_train, df_test

    def numpy_images(self, df_train, df_test):
        if not os.path.isfile('data/images_train.npz'):
            images_train = [self.image_edit0(path, 32) for path in df_train['path']]
            np.savez("data/images_train", images_train)
            array_train = np.asarray(images_train)
        else:
            images_train = np.load("data/images_train.npz")
            array_train = np.asarray(images_train['arr_0'])

        if not os.path.isfile('data/images_test.npz'):
            images_test = [self.image_edit0(path, 32) for path in df_test['path']]
            np.savez("data/images_test", images_test)
            array_train = np.asarray(images_train)
        else:
            images_test = np.load("data/images_test.npz")
            array_test = np.asarray(images_test['arr_0'])

        return array_train, array_test

    def normalize_images(self, X_train, X_test, array_test):
        X_normalized = self.normalize_scale(X_train)
        X_normalized_test = self.normalize_scale(X_test)
        test_normalized = self.normalize_scale(array_test)

        return X_normalized, X_normalized_test, test_normalized


if __name__ == '__main__':

    mod = Model()

    df_train, df_test = mod.proc_or_load()

    array_train, array_test = mod.numpy_images(df_train, df_test)

    X_train, X_test, y_train, y_test = train_test_split(array_train,
                                                        df_train['label'],
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=df_train['label'])

    X_normalized, X_normalized_test, test_normalized = mod.normalize_images(X_train, X_test, array_test)

    y_one_hot = mod.label_binarizer.fit_transform(y_train)
    y_one_hot_test = mod.label_binarizer.fit_transform(y_test)


    model = Sequential()
    model.add(Convolution2D(16, 4, 4, input_shape=(32, 32, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 4, 4, activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    # model.add(Convolution2D(128, 4, 4, activation="relu"))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(12, activation="softmax"))

    BATCH_SIZE = 8
    EPOCHS = 30
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    gen = ImageDataGenerator(
        rotation_range=360.,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=True
    )
    model.fit_generator(gen.flow(X_normalized, y_one_hot, batch_size=BATCH_SIZE),
                        steps_per_epoch=250,
                        epochs=EPOCHS,
                        verbose=1,
                        shuffle=True,
                        validation_data=(X_normalized_test, y_one_hot_test))

    earlyStopping = keras.callbacks.EarlyStopping(monitor='acc', patience=4, verbose=1, mode='auto')

    history = model.fit(X_normalized,
                        y_one_hot,
                        steps_per_epoch=100,
                        epochs=80,
                        verbose=1,
                        callbacks=[earlyStopping],
                        )

    preds = model.predict(X_normalized_test)
    predictions = mod.label_binarizer.inverse_transform(preds)
    f1_s = round(f1_score(y_test, predictions, average='micro'), 4)

    model_name = 'modelo_{}'.format(str(f1_s))
    model.save_weights('weights/{}.h5'.format(model_name))
    model.save('models/{}.h5'.format(model_name))

    model_load = load_model('models/{}.h5'.format(model_name))
    model_load.load_weights('weights/{}.h5'.format(model_name))
    test_score = model_load.evaluate(X_normalized_test, y_one_hot_test)
    print(test_score)

    mod.submit(model_load, test_normalized, 'f1s_{}.csv'.format(str(f1_s)), df_test)




