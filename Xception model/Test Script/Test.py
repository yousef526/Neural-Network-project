import cv2
import numpy as np
import os
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, Add
from keras.layers import SeparableConv2D, ReLU
from keras.layers import BatchNormalization, MaxPool2D
from keras.layers import GlobalAvgPool2D
from keras import Model
from keras.models import load_model
import keras, os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import math
import random

newmodel = load_model('../best_weight_2')

TEST_DIR = 'tuesday_test'


def load_images_test_from_folder(folder):
    images = []
    images_names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img, (299, 299))
        if img is not None:
            images.append([np.array(img_data)])
            images_names.append(filename)
    return images, images_names


test, images_names = load_images_test_from_folder(TEST_DIR)
X_test = np.array([i for i in test]).reshape((-1, 299, 299, 3))

from tensorflow import keras

predictions = newmodel.predict(X_test)
pred = []



for prediction in predictions:
    max_val = np.argmax(prediction)
    pred.append(max_val)

import csv

headers = ["image_name", "label"]
OutPut_list = []

for i in range(len(pred)):
    x = [images_names[i], pred[i]]
    OutPut_list.append(x)

with open("CS_H28.csv", "w", newline='') as Sport:
    student = csv.writer(Sport)
    student.writerow(headers)
    student.writerows(OutPut_list)
