#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import optimizers

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder



train_path = './train/'
train_truth = pd.read_csv('train.truth.csv')

images = []

for name in train_truth['fn']:
    img = cv2.imread(train_path + name)
    images.append(img)


train_images = np.array(images)
data_size, img_row, img_col, channels = train_images.shape

labels = train_truth['label']
categories = pd.unique(labels)


onehot_encoder = OneHotEncoder()
ohe_labels = onehot_encoder.fit_transform(labels.values.reshape((len(labels),1)))

model = Sequential()

model.add(Conv2D(5, kernel_size=3, padding='same', strides=(1, 1), activation='relu', input_shape=(img_row, img_col, channels)))
model.add(Conv2D(5, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
model.add(Dense(len(categories), activation='softmax'))

model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# sss = StratifiedShuffleSplit(n_splits=20)

# train_index, test_index = next(iter(sss.split(train_images, labels)))
# X_index, y_index = next(iter(sss.split(train_images[train_index], labels[train_index])))

early_stopping_monitor = EarlyStopping(patience=2)
# lrs = LearningRateScheduler(schedule=lambda epoch: 0.9 ** epoch)


model.fit(train_images, ohe_labels, validation_split=0.3, epochs=20,callbacks=[early_stopping_monitor])
# model.fit(train_images[test_index], ohe_labels[test_index], validation_split=0.3, epochs=10)


import glob

test_path= './test/'

test_images = []
names = []
for filename in glob.glob(test_path + '*' ):
    names.append(filename.split('/')[2])
    test_images.append(cv2.imread(filename))

test_images = np.array(test_images)
test_images.shape


pred = model.predict(test_images)

pred_decoded = onehot_encoder.inverse_transform(pred)

pred_decoded = list(pred_decoded)

pred_df = pd.DataFrame({'fn': names, 'label': pred_decoded})

pred_df.to_csv('test.preds.csv')

def correct_image(image, rotation):
    
    cols, rows, channels = image.shape
    angle = 0
    
    if rotation == 'rotated_left':
        angle = -90
    elif rotation == 'rotated_right':
        angle = 90
    
    elif rotation == 'upside_down':
        angle = 180
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_result = cv2.warpAffine(image,M,(cols,rows))
        
    return img_result

for i, img in enumerate(test_images):
    corrected_image = correct_image(img, pred_df.iloc[i]['label'])
    cv2.imwrite('results' + pred_df.iloc[i]['fn'] + '_corrected', corrected_image)
    