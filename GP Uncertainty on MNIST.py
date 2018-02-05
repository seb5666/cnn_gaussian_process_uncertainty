
# coding: utf-8

# In[3]:


from get_mnist_features import get_gp_mnist_data
from keras.datasets import mnist
import numpy as np
import scipy.io


# In[4]:


# %%time
# (mnist_train, X_train, y_train) ,(mnist_test, X_test, y_test) = get_gp_mnist_data('my_model.h5')
mnist_train = np.load('data/mnist_train.npy')
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
mnist_test = np.load('data/mnist_test.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')


# In[5]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # Train a GP on the features

# In[6]:


import gpflow
import tensorflow as tf

from sklearn.metrics import classification_report

from matplotlib import pyplot as plt


# In[7]:


num_classes = 10
num_features = 128


# In[14]:


get_ipython().run_cell_magic('time', '', "X_train = X_train.astype('float64').reshape(-1,num_features)\ny_train = y_train.astype('float64').reshape(-1,1)\nkernel = gpflow.kernels.Matern32(num_features) + gpflow.kernels.White(num_features, variance=0.01)\nlikelihood = gpflow.likelihoods.MultiClass(num_classes)\nZ=X_train[::5].copy()\nnum_latent=num_classes\nwhiten=True\nq_diag=True\nm = gpflow.models.SVGP(X_train, y_train, \n                       kern=kernel, \n                       likelihood=likelihood,\n                       Z=Z, \n                       num_latent=num_latent,\n                       whiten=whiten, \n                       q_diag=q_diag)\n\nm.kern.white.variance.trainable = False\nm.feature.trainable = False\nm.as_pandas_table()")


# In[15]:


get_ipython().run_cell_magic('time', '', 'opt = gpflow.train.ScipyOptimizer()\nopt.minimize(m)')


# In[16]:


get_ipython().run_cell_magic('time', '', "X_test = X_test.astype('float64').reshape(-1,num_features)\nprint(X_test.shape)\np, var = m.predict_y(X_test)")


# In[17]:


print(p.shape)


# In[18]:


predictions = np.argmax(p, axis=-1)
accuracy = len(np.argwhere(predictions == y_test))/len(predictions)
print(accuracy)
print(classification_report(y_test, predictions))


# In[19]:


incorrect_indices = np.argwhere(predictions != y_test)
correct_indices = np.argwhere(predictions == y_test)
print(len(incorrect_indices))


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
index = incorrect_indices[5]
print(var[index].squeeze())
plt.bar(np.arange(10), p[index].squeeze(), yerr=var[index].squeeze())
print("Correct class: {}".format(y_test[index]))
plt.figure()
plt.imshow(mnist_test[index].reshape(28,28),cmap =plt.cm.gray_r, interpolation = "nearest") 


# In[21]:


# %matplotlib inline
index = correct_indices[5496]
print(index)
print(p[index].squeeze())
print(var[index].squeeze())
plt.bar(np.arange(10), p[index].squeeze(), yerr=var[index].squeeze())
print("Correct class: {}".format(y_test[index]))
plt.figure()
plt.imshow(mnist_test[index].reshape(28,28),cmap =plt.cm.gray_r, interpolation = "nearest") 


# In[22]:


correct_indices_correct_classes = np.argmax(p[correct_indices], axis=-1).reshape(-1)
correct_indices_correct_var = temp[np.arange(len(correct_indices)), correct_indices_correct_classes]
np.argmax(correct_indices_correct_var)


# ## Noisy mnist

# In[99]:


def load_data():
    rows, cols = 28, 28
    nb_classes = 10

    DATA_DIR = 'data/mnist-with-awgn.mat'
    mat = scipy.io.loadmat(DATA_DIR)

    X = mat['images']
    Y = mat['labels']

    # Move last column to front
    X = np.rollaxis(X, 2)

    # Reshape and format input
    X = X.reshape(X.shape[0], rows, cols, 1)
    X = X.astype('float32')
    X /= 255.0

    # Hot encoding
    Y = Y.astype(int)
    Y = np_utils.to_categorical(Y, nb_classes)

    # Divide into test and train sets
    perm = np.random.permutation(X.shape[0])

    train_size = 13000

    X_train = X[perm[:train_size]]
    X_test = X[perm[train_size:]]

    Y_train = Y[perm[:train_size]]
    Y_test = Y[perm[train_size:]]

    return (X_train, Y_train, X_test, Y_test)


# In[102]:


(X_train, Y_train, X_test, Y_test) = load_data()

