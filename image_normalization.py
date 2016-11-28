import numpy as np
import tensorflow as tf
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import cv2
# import time
# from datetime import timedelta

# Reload the data
# pickle_file = 'X_in_yuv_colorspace.pickle'
# with open(pickle_file, 'rb') as f:
#     pickle_data = pickle.load(f)
#     X_train_yuv = pickle_data['X_train_yuv']
#     X_test_yuv = pickle_data['X_test_yuv']
#     y_train = pickle_data['y_train']
#     y_test = pickle_data['y_test']
#     del pickle_data  # Free up memory

# Load pickled data
import pickle
import os
# TODO: fill this in based on where you saved the training and testing data
training_file = './train.p'
testing_file = './test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train_yuv, y_train = train['features'], train['labels']
X_test_yuv, y_test = test['features'], test['labels']
print('Data and modules loaded.')
del train, test

train_rows = X_train_yuv.shape[0]
test_rows = X_test_yuv.shape[0]
# Pre-processing
X_train_yuv = tf.Variable(X_train_yuv)
X_test_yuv = tf.Variable(X_test_yuv)

session = tf.Session()
session.run(tf.initialize_all_variables())

X_train_yuv = tf.image.rgb_to_grayscale(X_train_yuv)
X_test_yuv = tf.image.rgb_to_grayscale(X_test_yuv)
X_train_yuv = session.run(X_train_yuv)
X_test_yuv = session.run(X_test_yuv)

print('start')

# todo this whitening operation is really slow. find some fast ways.
for i in range(train_rows):
    X_train_yuv[i] = session.run(tf.image.per_image_whitening(X_train_yuv[i]))

whiten_train = 'whiten_train.p'
if not os.path.isfile(whiten_train):
    print('Saving data to pickle file...')
    try:
        with open(whiten_train, 'wb') as pfile:
            pickle.dump(
                {
                    'X_train_yuv': X_train_yuv,
                    'y_train': y_train,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', whiten_train, ':', e)
        raise

print('X_train_yuv saved in ', whiten_train)

for j in range(test_rows):
    X_test_yuv[j] = session.run(tf.image.per_image_whitening(X_test_yuv[j]))

print('X_train type ', type(X_train_yuv))
print('X_test type: ', type(X_test_yuv))
print(X_train_yuv.shape)
print(X_test_yuv.shape) # Save the X_train_yuv for easy access

pickle_file = 'gray_normalized.p'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open('gray_normalized.p', 'wb') as pfile:
            pickle.dump(
                {
                    'X_train_yuv': X_train_yuv,
                    'X_test_yuv': X_test_yuv,
                    'y_train': y_train,
                    'y_test': y_test,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')
print(y_train.shape)
print(y_test.shape)