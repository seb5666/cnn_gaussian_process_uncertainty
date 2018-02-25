from get_mnist_features import get_gp_mnist_data
from keras.datasets import mnist
import numpy as np
import scipy.io

import gpflow
import tensorflow as tf

from sklearn.metrics import classification_report

from matplotlib import pyplot as plt

import time

load_data_from_disk = True
restore_model = True
save_model = False

if load_data_from_disk:
    
    mnist_train = np.load('data/mnist_train.npy')
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    mnist_test = np.load('data/mnist_test.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    print("Loaded data")
else:
    (mnist_train, y_train), (mnist_test, y_test) = mnist.load_data()
    
    mnist_train = mnist_train.astype('float32')
    mnist_test = mnist_test.astype('float32')

    mnist_train /= 255
    mnist_test /= 255
    
    mnist_train = mnist_train.reshape(mnist_train.shape[0], 28, 28, 1)
    mnist_test = mnist_test.reshape(mnist_test.shape[0], 28, 28, 1)
    
    X_train = get_gp_mnist_data('my_model.h5', mnist_train)
    X_test = get_gp_mnist_data('my_model.h5', mnist_test)
    
    np.save('data/mnist_train.npy', mnist_train)
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/mnist_test.npy', mnist_test)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    print("Saved data")

num_classes = 10
num_features = 128

X_train = X_train.astype('float64').reshape(-1,num_features)[:59000]
y_train = y_train.astype('float64').reshape(-1,1)[:59000]
X_test = X_test.astype('float64').reshape(-1,num_features)

kernel = gpflow.kernels.Matern32(num_features) + gpflow.kernels.White(num_features, variance=0.01)
likelihood = gpflow.likelihoods.MultiClass(num_classes)
Z=X_train[::1000].copy()
num_latent=num_classes
whiten=True
q_diag=True
m = gpflow.models.SVGP(X_train, y_train,
                       kern=kernel,
                       likelihood=likelihood,
                       Z=Z,
                       num_latent=num_latent,
                       whiten=whiten,
                       q_diag=q_diag)

m.kern.white.variance.trainable = False
m.feature.trainable = False

if restore_model:
    saver = tf.train.Saver()
    sess = gpflow.session_manager.get_default_session()
    saver.restore(sess, "models/gp.ckpt")
    print("restored model")
else:
    start_time = time.time()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    print("Optimisation done in: {:0.4f}s".format(time.time() -start_time))

p, var = m.predict_y(X_test)
print("Predictions for test data done")
print(p.shape)

if save_model:
    saver = tf.train.Saver()
    sess = gpflow.session_manager.get_default_session()
    saver.save(sess, "models/gp.ckpt")
    print("saved vars")


predictions = np.argmax(p, axis=-1)
accuracy = len(np.argwhere(predictions == y_test))/len(predictions)
print("Accuracy: {}".format(accuracy))

incorrect_indices = np.argwhere(predictions != y_test)
correct_indices = np.argwhere(predictions == y_test)
print("Number of wrong classifications: {}".format(len(incorrect_indices)))

