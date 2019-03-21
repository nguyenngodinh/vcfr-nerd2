 from __future__ import print_function
from __future__ import division
import os
import numpy as np
from keras import backend as K
from keras.utils.data_utils import get_file 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import clone_model
from keras.models import load_model
import os
import tensorflow
import keras
from keras.preprocessing import image
from sklearn import svm
import pickle
import shutil
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply, ZeroPadding2D, Convolution2D, Dropout
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K

from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Model
from keras import layers
import csv
from keras.models import Sequential

#declare model 

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten(name='flatten'))
model.add(Activation('softmax'))

model.load_weights(filepath='/home/dev/aivivn/vgg_face_weights.h5')
model.summary()
extractor = Model(model.input, model.get_layer('flatten').output)
extractor.summary()



#get image and extract feature 
debugmode1 = False
imagearrs = []
labels = []
stats = [0]*1000
ifolder = '/home/dev/vn_celeb_face_recognition/train'
ftrain = open('/home/dev/vn_celeb_face_recognition/train.csv', 'r')
for x in ftrain:
    if debugmode1:
        print (x)
    path, iid = x.split(',')
    if path.endswith('.png') or path.endswith('.jpg'):
        if debugmode1:
            print(path)
            print(iid)
#         extract feature of image and 
        imgpath = os.path.join(ifolder, path)
        stats[int(iid)] += 1
        if debugmode1:
            print(imgpath)
        img = image.load_img(imgpath, target_size=(224, 224))
        imgarr = image.img_to_array(img)
        labels.append(int(iid))
        imagearrs.append(imgarr)
if debugmode1:
    print(len(imagearrs))
    print(len(labels))
    print(labels)
imagearrs = np.asarray(imagearrs)
labels = np.asarray(labels)
features = extractor.predict(imagearrs)
if debugmode1:
    print(labels.shape)
    print(features.shape)
    
    
#train svm models 
import pickle
from sklearn.model_selection import KFold, cross_val_score
debugmode2 = True 
kfold = KFold(n_splits=5)
svm_model_root = '/home/dev/svmmodels'
gamma = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] 
c = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] 
for i in gamma:
    for j in c:
        svmname = os.path.join(svm_model_root, 'svm-' + str(i) + '-' + str(j) + '.sav')
        if debugmode2:
            print(svmname)
        svm_model = svm.SVC(kernel='rbf', gamma = i, C=j, probability=True)
        [svm_model.fit(features[train], labels[train]).score(features[test], labels[test]) for train, test in kfold.split(features)]
        check = svm_model.predict(features)
        check_res = check == labels
        acc = (np.count_nonzero(check_res))/check_res.size * 100
        if debugmode2:
            print(i)
            print(acc)
            print('\n')
        pickle.dump(svm_model, open(svmname, 'wb'), protocol = 2)


#run svm models and save result 
import csv
from keras.preprocessing import image
debugmode3 = True

test_fol = '/home/dev/vn_celeb_face_recognition/test'
res_root = '/home/dev/results'

for model in os.listdir(svm_model_root):
    svm_model = pickle.load(open(os.path.join(svm_model_root, model), 'rb'))
    if debugmode3:
        print (open(os.path.join(svm_model_root, model)))
    with open(os.path.join(res_root, 'res' + model.replace('.sav', '') + '.csv') , 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_NONE)
        writer.writerow(['image','label'])
        if debugmode3: 
            print(os.path.join(res_root, 'res' + model + '.csv'))
        for imgname in os.listdir(test_fol):
            if imgname.endswith('.png'):
                img = image.load_img(os.path.join(test_fol, imgname), target_size=(224, 224))
                if debugmode3:
                    print(os.path.join(test_fol, imgname))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                feat = extractor.predict(img)
                iid = svm_model.predict_proba(feat)
                iid = np.squeeze(iid)
                iid = np.insert(iid, iid.shape[0], 0.0005)
                top5 = np.argsort(iid)[-5:][::-1]
                print(top5)
                print(iid[top5[0]], iid[top5[1]], iid[top5[2]], iid[top5[3]], iid[top5[4]])
                writer.writerow([imgname, ' '.join([str(top5[0]), str(top5[1]), str(top5[2]), 
                                                    str(top5[3]), str(top5[4])])])
