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
from xgboost import XGBClassifier
import xgboost as xgb
import time
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore")

class stack_nn(object):

    def __init__(self):
        self.rfuncs = Model()

        self.m1 = self.rfuncs.load_m('modelo_0.8716')
        self.m2 = self.rfuncs.load_m('modelo_0.8674')
        self.m3 = self.rfuncs.load_m('modelo_0.8274')
        self.m4 = self.rfuncs.load_m('modelo_0.7716')
        self.m5 = self.rfuncs.load_m('modelo_0.7021')
        self.m6 = self.rfuncs.load_m('modelo_0.8695')
        self.m7 = self.rfuncs.load_m('modelo_0.8063')
        self.m8 = self.rfuncs.load_m('modelo_0.8032')


        self.label_binarizer = LabelBinarizer()

        self.df_train, self.df_test = self.rfuncs.proc_or_load()

        array_train, array_test = self.rfuncs.numpy_images(self.df_train, self.df_test)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(array_train,
                                                            self.df_train['label'],
                                                            test_size=0.2,
                                                            random_state=42,
                                                            stratify=self.df_train['label'])

        self.X_normalized, self.X_normalized_test, self.test_normalized = self.rfuncs.normalize_images(array_train,
                                                                                                       self.X_test,
                                                                                                       array_test)

    def model_preds(self, model):
        preds = model.predict(self.X_normalized)
        predictions = self.label_binarizer.inverse_transform(preds)
        return predictions



if __name__ == '__main__':
    stacking = stack_nn()

    # print('train/test split')
    # X_train, X_test, y_train, y_test = train_test_split(stacking.X_normalized,
    #                                                     stacking.df_train['label'],
    #                                                     test_size=0.2,
    #                                                     random_state=42,
    #                                                     stratify=stacking.df_train['label'])

    print('one hot label binarizer')
    stacking.label_binarizer.fit(stacking.df_train['label'])
    # y_one_hot = stacking.label_binarizer.fit_transform(y_train)
    # y_one_hot_test = stacking.label_binarizer.fit_transform(y_test)

    print('predictions')
    m1_preds = stacking.model_preds(stacking.m1)
    m2_preds = stacking.model_preds(stacking.m2)
    m3_preds = stacking.model_preds(stacking.m3)
    # m4_preds = stacking.model_preds(stacking.m4)
    m5_preds = stacking.model_preds(stacking.m5)
    m6_preds = stacking.model_preds(stacking.m6)
    m7_preds = stacking.model_preds(stacking.m7)
    m8_preds = stacking.model_preds(stacking.m8)

    print('converting predictions to pandas DF')
    full_preds = {"m1": m1_preds,
                  "m2": m2_preds,
                  "m3": m3_preds,
                  # "m4": m4_preds,
                  "m5": m5_preds,
                  "m6": m6_preds,
                  "m7": m7_preds,
                  "m8": m8_preds,
                  "Y": stacking.df_train['label']
                  }

    df_preds = pd.DataFrame(full_preds, index=None)

    # print(df_preds.head(10))

    print('train/test split')
    X_train, X_test, y_train, y_test = train_test_split(df_preds.loc[:, df_preds.columns != 'Y'],
                                                        df_preds['Y'],
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=df_preds['Y'])

    fit_dict_xgbc = {"eval_set": [(X_train, y_train)],
                     "early_stopping_rounds": 5,
                     "verbose": True,
                     "eval_metric": "mlogloss",
                     }

    parameters_xgbc = {"learning_rate": [0.05],
                       "max_depth": [10, 20],
                       "n_estimators": [500, 600],
                       # "max_deta_step": [1, 3, 5]
                       }

    xgboost_estimator = XGBClassifier(nthread=4,
                                  seed=42,
                                  subsample=0.8,
                                  colsample_bytree=0.6,
                                  colsample_bylevel=0.7,
                                  )

    xgboost_model = GridSearchCV(estimator=xgboost_estimator,
                                 param_grid=parameters_xgbc,
                                 n_jobs=4,
                                 cv=5,
                                 fit_params=fit_dict_xgbc,
                                 verbose=10,
                                 )
    tmp = time.time()
    xgboost_model.fit(X_train, y_train,
                      # evals=[(dtest, 'test')],
                      # evals_result=gpu_res
                      )
    print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))


    stacking_preds = xgboost_model.predict(X_test)

    f1_s = round(f1_score(y_test, stacking_preds, average='micro'), 4)
    print('\nf1_score for xgboost stacking predictions:',f1_s)

    joblib.dump(xgboost_model.best_estimator_, 'models/model_stacking_{}.pkl'.format(f1_s))
