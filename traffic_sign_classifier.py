import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import time
from datetime import timedelta

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
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
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

# Pre-processing
X_train_yuv = tf.Variable(X_train_yuv)
X_test_yuv = tf.Variable(X_test_yuv)

session = tf.Session()
session.run(tf.initialize_all_variables())

X_train_yuv = tf.image.rgb_to_grayscale(X_train_yuv)
X_test_yuv = tf.image.rgb_to_grayscale(X_test_yuv)

X_train_yuv = session.run(X_train_yuv)
X_test_yuv = session.run(X_test_yuv)

# shuffle them 5 times
for i in range(5):
    X_train_yuv, y_train = shuffle(X_train_yuv, y_train, random_state=i)
    X_test_yuv, y_test = shuffle(X_test_yuv, y_test, random_state=i)

print('X_train type ', type(X_train_yuv))
print('X_test type: ', type(X_test_yuv))
print(X_train_yuv.shape)
print(X_test_yuv.shape)
print(y_train.shape)
print(y_test.shape)

img_num = X_train_yuv.shape[0]
img_height = X_train_yuv.shape[1]
img_width = X_train_yuv.shape[2]
num_channel = X_train_yuv.shape[3]
n_classes = len(set(y_train))

x = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 1], name='x')

# Todo may need to be changed
y_true = tf.placeholder(tf.int64, shape=[None], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.
fc_size = 128             # Number of neurons in fully-connected layer.

from model_arch import *
layer_conv1, weights_conv1 = new_conv_layer(input=x, num_input_channels=num_channel, filter_size=filter_size1,
                                            num_filters=num_filters1, use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1,
                                            filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=n_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# y_true is in 1-d shape, sparse_softmax_cross_entroy_with_logits can do auto one-hot encoding.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Performance measure
correct_prediction = tf.equal(y_pred_cls, y_true) # if it is processed inside the session
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.initialize_all_variables())

train_batch_size = 64
total_iterations = 0


def optimize(num_iterations):
    global total_iterations
    start_time = time.time()
    batch_index = np.array(range(train_batch_size))

    for i in range(total_iterations, total_iterations + num_iterations):
        # x_batch, y_true_batch = data.train.next_batch(train_batch_size) # numpy array
        try:
            x_batch = X_train_yuv[batch_index]
            y_true_batch = y_train[batch_index]
            batch_index += train_batch_size
            # re-count batches from index 0
            batch_index = batch_index % img_num
        except IndexError:
            print('lance out of index: ', batch_index)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = X_test_yuv[incorrect]

    cls_pred = cls_pred[incorrect]

    cls_true = y_test[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_pred):
    cls_true = y_test
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, range(n_classes))
    plt.yticks(tick_marks, range(n_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

test_batch_size = 256


def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    num_test = len(y_test)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = X_test_yuv[i:j]
        labels = y_test[i:j]
        # labels should be
        feed_dict = {x: images, y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    cls_true = y_test

    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

print(y_test[0])

print(y_train)

print_test_accuracy()

optimize(num_iterations=1)

print_test_accuracy()

optimize(num_iterations=99)  # We already performed 1 iteration above.

print_test_accuracy(show_example_errors=True)

optimize(num_iterations=900)  # We performed 100 iterations above.

print_test_accuracy(show_example_errors=True)

#
# print_test_accuracy()
#
# optimize(num_iterations=99)  # We already performed 1 iteration above.
#
# print_test_accuracy(show_example_errors=True)




