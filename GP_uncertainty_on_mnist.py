from get_mnist_features import get_gp_mnist_data
from keras.datasets import mnist
import numpy as np
import scipy.io

import gpflow
import tensorflow as tf

from sklearn.metrics import classification_report

from matplotlib import pyplot as plt


(mnist_train, X_train, y_train) ,(mnist_test, X_test, y_test) = get_gp_mnist_data('my_model.h5')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

num_classes = 10
num_features = 128

X_train = X_train.astype('float64').reshape(-1,num_features)
y_train = y_train.astype('float64').reshape(-1,1)
kernel = gpflow.kernels.Matern32(num_features) + gpflow.kernels.White(num_features, variance=0.01)
likelihood = gpflow.likelihoods.MultiClass(num_classes)
Z=X_train[::5].copy()
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
print(m.as_pandas_table())

opt = gpflow.train.ScipyOptimizer()\nopt.minimize(m)
print("Optimisation done")
X_test = X_test.astype('float64').reshape(-1,num_features)
p, var = m.predict_y(X_test)
print("Predictions for test data done")
print(p.shape)

predictions = np.argmax(p, axis=-1)
accuracy = len(np.argwhere(predictions == y_test))/len(predictions)
print("Accuracy: {}".format(accuracy))
print(classification_report(y_test, predictions))

incorrect_indices = np.argwhere(predictions != y_test)
correct_indices = np.argwhere(predictions == y_test)
print("Number of wrong classifications: {}".format(len(incorrect_indices)))

#index = incorrect_indices[5]
#print(var[index].squeeze())
#plt.bar(np.arange(10), p[index].squeeze(), yerr=var[index].squeeze())
#print("Correct class: {}".format(y_test[index]))
#plt.figure()
#plt.imshow(mnist_test[index].reshape(28,28),cmap =plt.cm.gray_r, interpolation = "nearest")
## %matplotlib inline
#index = correct_indices[5496]
#print(index)
#print(p[index].squeeze())
#print(var[index].squeeze())
#plt.bar(np.arange(10), p[index].squeeze(), yerr=var[index].squeeze())
#print("Correct class: {}".format(y_test[index]))
#plt.figure()
#plt.imshow(mnist_test[index].reshape(28,28),cmap =plt.cm.gray_r, interpolation = "nearest")
#
#
## In[22]:
#
#
#correct_indices_correct_classes = np.argmax(p[correct_indices], axis=-1).reshape(-1)
#correct_indices_correct_var = temp[np.arange(len(correct_indices)), correct_indices_correct_classes]
#np.argmax(correct_indices_correct_var)

