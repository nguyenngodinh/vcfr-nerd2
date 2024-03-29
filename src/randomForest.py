# -*- coding: utf-8 -*-
"""aivivn_face_recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hUfor0nK0cQCTDaenS3it_rIzskqKRAz
"""

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

!apt autoremove
!apt update && apt install -y libsm6 libxext6
!pip install gdown
!gdown https://drive.google.com/uc?id=1kpxjaz3pIMrAhEjm7hJxcBsxKNhfl8t2&export=download
!unzip vn_celeb_face_recognition.zip
!rm vn_celeb_face_recognition.zip
!clear && ls

from google.colab import drive
drive.mount('/content/gdrive')
!cd "/content/gdrive/My Drive"

from keras.models import load_model
from keras.models import Model
from keras.preprocessing import image
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import metrics


debugMode = True
debugNumTrain = 4720

ft_model = load_model(filepath='/content/gdrive/My Drive/face_finetune_model.h5')
extractor = Model(ft_model.input, ft_model.get_layer('flatten').output)
if debugMode:
  extractor.summary()

trainDir = '/content/train/'
testDir = '/content/test/'
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('sample_submission.csv')
if debugMode:
  
  print(trainData.head(min(5, debugNumTrain)))
  print(testData.head(min(5, debugNumTrain)))

imgarrs = []

if debugMode:
  imgSeries = trainData.image.head(debugNumTrain)
else:
  imgSeries = trainData.image
  
for imgName in imgSeries:
  imgPath = trainDir + imgName
  img = image.load_img(imgPath, target_size=(224, 224))
  imgarr = image.img_to_array(img)
  imgarrs.append(imgarr)
imgarrs = np.asarray(imgarrs)
if debugMode:
  print(imgarrs.shape)

features = extractor.predict(imgarrs)
del imgarrs
if debugMode:
  print(type(features))
  print(features.shape)

lbls = []
if debugMode:
  lblSeries = trainData.label.head(debugNumTrain)
else:
  lblSeries = trainData.label

for lbl in lblSeries:
  lbls.append(lbl)
lbls = np.asarray(lbls)
if debugMode:
  print(lbls)

# max_depth = [int(x) for x in np.linspace(10, 110, num = 3)]
# max_depth.append(None)

# random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 120, num = 4)],
#               'max_features': ['sqrt', 'log2', None],
#               'max_depth': max_depth,
#               'min_samples_split': [int(x) for x in np.linspace(start = 2, stop = 12, num = 3)],
#               'min_samples_leaf': [int(x) for x in np.linspace(start = 1, stop = 5, num = 3)],
#               'bootstrap': [True],
#               'n_jobs':[-1]}

# rf_grid = RandomizedSearchCV(estimator = RandomForestClassifier(random_state=42), 
#                          param_distributions = random_grid, cv = 3, n_jobs = -1, verbose = 2)
# rf_grid.fit(features, lbls)

# print(rf_grid.best_params_)

# rf = rf_grid.best_estimator_

rf = RandomForestClassifier(n_estimators = 300, max_features = 'sqrt', max_depth = 110, 
                            bootstrap = True, random_state = 42, n_jobs = -1)
rf.fit(features, lbls)

for idx in range(testData.shape[0]):
  imgPath = testDir + testData.loc[idx, 'image']
  img = image.load_img(imgPath, target_size=(224, 224))
  imgarr = image.img_to_array(img)
  imgarrs = np.asarray([imgarr])
  features_test = extractor.predict(imgarrs)
  del imgarrs
  iid = rf.predict_proba(features_test)
  iid = np.squeeze(iid) 
  iid = np.insert(iid, iid.shape[0], 0.015)
  top5 = np.argsort(iid)[-5:][::-1]
  #print(top5)
  #print(iid[top5[0]], iid[top5[1]], iid[top5[2]], iid[top5[3]], iid[top5[4]])
  testData.loc[idx, 'label'] = ' '.join([str(top5[0]), str(top5[1]), str(top5[2]), 
                                                    str(top5[3]), str(top5[4])])

file_out = 'random_forest_300.csv'
testData.to_csv(file_out, index = False)

!cp random_forest_300.csv "/content/gdrive/My Drive/"

#!ls /content/train
#help(rf)
#!head random_forest_best.csv