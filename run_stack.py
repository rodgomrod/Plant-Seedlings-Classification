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
from run import *

class stack_nn(object):

    def __init__(self):
        rfuncs = Model()

        # self.m1 = rfuncs.load_m('modelo_0.8716')
        self.m2 = rfuncs.load_m('modelo_0.8674')
        self.m2 = rfuncs.load_m('modelo_0.7716')
        self.m2 = rfuncs.load_m('modelo_0.7021')

    def model_preds(self):
        pass








if __name__ == '__main__':
    stack_nn()
    # pass