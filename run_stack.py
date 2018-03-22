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
import sys

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

    def model_preds(self, model, X):
        preds = model.predict(X)
        predictions = self.label_binarizer.inverse_transform(preds)
        return predictions

    def submit_stack(self, model, data, sub_name):
        df_test = pd.read_csv('data/df_data/test.csv')
        path_test = [df_test['path'][i].split('/')[-1] for i in range(len(self.df_test['path']))]
        pr1 = model.predict(data)
        names_pr = [self.rfuncs.number_to_names[str(i)] for i in pr1]
        submiss = pd.DataFrame({'file': path_test, 'species': names_pr})
        submiss.to_csv('submissions/{}'.format(sub_name), index=None)



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
    m1_preds = stacking.model_preds(stacking.m1, stacking.X_normalized)
    m2_preds = stacking.model_preds(stacking.m2, stacking.X_normalized)
    m3_preds = stacking.model_preds(stacking.m3, stacking.X_normalized)
    # m4_preds = stacking.model_preds(stacking.m4)
    m5_preds = stacking.model_preds(stacking.m5, stacking.X_normalized)
    m6_preds = stacking.model_preds(stacking.m6, stacking.X_normalized)
    m7_preds = stacking.model_preds(stacking.m7, stacking.X_normalized)
    m8_preds = stacking.model_preds(stacking.m8, stacking.X_normalized)

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

    print('train/test split')
    X_train, X_test, y_train, y_test = train_test_split(df_preds.loc[:, df_preds.columns != 'Y'],
                                                        df_preds['Y'],
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=df_preds['Y'])

    # TRAINING ON CPU

    # fit_dict_xgbc = {"eval_set": [(X_train, y_train)],
    #                  "early_stopping_rounds": 5,
    #                  "verbose": True,
    #                  "eval_metric": "mlogloss",
    #                  }
    #
    #
    # parameters_xgbc = {"learning_rate": [0.05],
    #                    "max_depth": [10, 20],
    #                    "n_estimators": [500, 600],
    #                    }
    #
    #
    # xgboost_estimator = XGBClassifier(nthread=4,
    #                               seed=42,
    #                               subsample=0.8,
    #                               colsample_bytree=0.6,
    #                               colsample_bylevel=0.7,
    #                               )
    #
    # xgboost_model = GridSearchCV(estimator=xgboost_estimator,
    #                              param_grid=parameters_xgbc,
    #                              n_jobs=4,
    #                              cv=5,
    #                              fit_params=fit_dict_xgbc,
    #                              verbose=10,
    #                              )
    # tmp = time.time()
    # xgboost_model.fit(X_train, y_train,
    #                   # evals=[(dtest, 'test')],
    #                   # evals_result=gpu_res
    #                   )
    # print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))


    # TRAINING ON GPU

    # Leave most parameters as default
    param = {'objective': 'multi:softmax',  # Specify multiclass classification
             'num_class': 12,  # Number of possible output classes
             'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
             'verbose': False,
             # 'gpu_id': [0],
             'silent': 1,
             'predictor': 'gpu_predictor',
             # 'n_jobs': -1,
             "learning_rate": 0.05,
             "max_depth": 20,
             "n_estimators": 600,
             'subsample': 0.8,
             "colsample_bytree": 0.6,
             "colsample_bylevel": 0.7,
             }

    # Convert input data from numpy to XGBoost format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    gpu_res = {}  # Store accuracy result
    tmp = time.time()
    # Train model
    xgboost_model = XGBClassifier(**param)
    xgboost_model.fit(X_train, y_train)
    print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))



    stacking_preds = xgboost_model.predict(X_test)

    f1_s = round(f1_score(y_test, stacking_preds, average='micro'), 4)
    print('\nf1_score for xgboost stacking predictions:',f1_s)

    joblib.dump(xgboost_model, 'models/model_stacking_GPU_{}.pkl'.format(f1_s))
    # joblib.dump(xgboost_model.best_estimator_, 'models/model_stacking_CPU_{}.pkl'.format(f1_s))

    xgboost_model = joblib.load('models/model_stacking_GPU_0.8716.pkl')

    print('submission out')

    m1_preds_test = stacking.model_preds(stacking.m1, stacking.test_normalized)
    m2_preds_test = stacking.model_preds(stacking.m2, stacking.test_normalized)
    m3_preds_test = stacking.model_preds(stacking.m3, stacking.test_normalized)
    m5_preds_test = stacking.model_preds(stacking.m5, stacking.test_normalized)
    m6_preds_test = stacking.model_preds(stacking.m6, stacking.test_normalized)
    m7_preds_test = stacking.model_preds(stacking.m7, stacking.test_normalized)
    m8_preds_test = stacking.model_preds(stacking.m8, stacking.test_normalized)

    print('converting predictions tests to pandas DF')
    full_preds_test = {"m1": m1_preds_test,
                  "m2": m2_preds_test,
                  "m3": m3_preds_test,
                  "m5": m5_preds_test,
                  "m6": m6_preds_test,
                  "m7": m7_preds_test,
                  "m8": m8_preds_test,
                  }


    df_preds_test = pd.DataFrame(full_preds_test, index=None)
    stacking.submit_stack(xgboost_model,
                          df_preds_test,
                          'f1s_{}.csv'.format(str(f1_s)),
                          )
