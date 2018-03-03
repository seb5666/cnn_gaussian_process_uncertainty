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
<<<<<<< Updated upstream
restore_model = True
=======
restore_model = False
>>>>>>> Stashed changes
save_model = False

if load_data_from_disk:
    mnist_train = np.load('data/mnist_train.npy')
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')

    mnist_test = np.load('data/mnist_test.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    # Load awgn created images
    awgn_images = np.load('data/awgn_images.npy')
    awgn_X = np.load('data/awgn_X.npy')
    awgn_y = np.load('data/awgn_y.npy')

    # Load n-MNIST
    (noisy_test, noisy_X_test, noisy_y_test) = np.load('data/noisy_mnist_test.npy'), np.load('data/noisy_X_test.npy'), np.load('data/noisy_y_test.npy')
    (blur_test, blur_X_test, blur_y_test) = np.load('data/blur_mnist_test.npy'), np.load('data/blur_X_test.npy'), np.load('data/blur_y_test.npy')

    (contrast_test, contrast_X_test, contrast_y_test) = np.load('data/contrast_mnist_test.npy'), np.load('data/contrast_X_test.npy') ,np.load('data/contrast_y_test.npy')

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
    
    # Create AWGN images and save
    num_images = 10
    indexes = np.arange(100)
    awgn_images = []
    for index in indexes:
        print(index)
        noise = np.random.randn(28, 28, 1)
        image = mnist_test[index]
        
        noisy_imgs = np.concatenate([(image + i/(num_images-1) * noise).reshape(1, 28, 28, 1) for i in range(num_images)]) 
        awgn_images.append(noisy_imgs)
    
    awgn_images = np.concatenate(awgn_images) 
    awgn_X = get_gp_mnist_data('my_model.h5', awgn_images)
    print(awgn_images.shape)
    print(awgn_X.shape)
    
    np.save('data/awgn_images.npy', awgn_images)
    np.save('data/awgn_X.npy', awgn_X)
    np.save('data/awgn_y.npy', y_test[indexes])
    

    # Load n-MNIST data and save i
    def load_data(DATA_DIR):
        rows, cols = 28, 28
        nb_classes = 10
        mat = scipy.io.loadmat(DATA_DIR)
        test = mat['test_x']
        Y_test = np.argmax(mat['test_y'], axis=-1)

        test = test.reshape(-1, rows, cols, 1)
        #Renormalize images
        test = test.astype('float32')
        test /= 255.0

        X_test = get_gp_mnist_data('my_model.h5', test) 

        return (test, X_test, Y_test)

    (noisy_test, noisy_X_test, noisy_y_test) = load_data('data/mnist-with-awgn.mat')
    np.save('data/noisy_mnist_test.npy', noisy_test)
    np.save('data/noisy_X_test.npy', noisy_X_test)
    np.save('data/noisy_y_test.npy', noisy_y_test)
    (blur_test, blur_X_test, blur_y_test) = load_data('data/mnist-with-motion-blur.mat')
    np.save('data/blur_mnist_test.npy', blur_test)
    np.save('data/blur_X_test.npy', blur_X_test)
    np.save('data/blur_y_test.npy', blur_y_test)
    (contrast_test, contrast_X_test, contrast_y_test) = load_data('data/mnist-with-reduced-contrast-and-awgn.mat')
    np.save('data/contrast_mnist_test.npy', contrast_test)
    np.save('data/contrast_X_test.npy', contrast_X_test)
    np.save('data/contrast_y_test.npy', contrast_y_test)

    print("Saved data")

num_classes = 10
num_features = 128

X_train = X_train.astype('float64').reshape(-1,num_features)[:59000]
y_train = y_train.astype('float64').reshape(-1,1)[:59000]
X_test = X_test.astype('float64').reshape(-1,num_features)

kernel = gpflow.kernels.Matern32(num_features) + gpflow.kernels.White(num_features, variance=0.01)
likelihood = gpflow.likelihoods.MultiClass(num_classes)
Z=X_train[::100].copy()
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

if save_model:
    print("Saving model")
    saver = tf.train.Saver()
    sess = gpflow.session_manager.get_default_session()
    saver.save(sess, "models/gp.ckpt")
    print("saved vars")

<<<<<<< Updated upstream
if save_model:
    saver = tf.train.Saver()
    sess = gpflow.session_manager.get_default_session()
    saver.save(sess, "models/gp.ckpt")
    print("saved vars")
=======

p, var = m.predict_y(X_test)
np.save("predictions/X_test.npy", [p, var])
>>>>>>> Stashed changes

print("Predictions for test data done")
print(p.shape)

predictions = np.argmax(p, axis=-1)
accuracy = len(np.argwhere(predictions == y_test))/len(predictions)
print("Accuracy: {}".format(accuracy))

## n-MNIST predictions 
#p2, var2 = m.predict_y(X_train)
#np.save("predictions/X_train.npy", [p2, var2])
#
#noisy_p, noisy_var = m.predict_y(noisy_X_test)
#np.save("predictions/noisy_test.npy", [noisy_p, noisy_var])
#noisy_predictions = np.argmax(noisy_p, axis=-1)
#print(len(np.argwhere(noisy_predictions == noisy_y_test))/len(noisy_predictions))
#
#blur_p, blur_var = m.predict_y(blur_X_test)
#np.save("predictions/blur_test.npy", [blur_p, blur_var])
#blur_predictions = np.argmax(blur_p, axis=-1)
#print(len(np.argwhere(blur_predictions == blur_y_test))/len(blur_predictions))
#
#contrast_p, contrast_var = m.predict_y(contrast_X_test)
#np.save("predictions/contrast_test.npy", [contrast_p, contrast_var])
#contrast_predictions = np.argmax(contrast_p, axis=-1)
#print(len(np.argwhere(contrast_predictions == contrast_y_test))/len(contrast_predictions))

## AWGN image
awgn_predictions = []
num_images = 10
indexes = np.arange(100)
awgn_X = awgn_X.reshape(100, num_images, -1)
for index in indexes:
    noisy_X = awgn_X[index]
    noisy_p, noisy_var = m.predict_y(noisy_X)
    awgn_predictions.append((noisy_p, noisy_var)) 
np.save("predictions/awgn.npy", awgn_predictions)

exit()


incorrect_indices = np.argwhere(predictions != y_test)
correct_indices = np.argwhere(predictions == y_test)
print("Number of wrong classifications: {}".format(len(incorrect_indices)))

# Use uncertainty to hold out if possible
print(p.shape)
print(var.shape)
sorted_indexes = np.argsort(p, axis=-1)

highest_p = p[np.arange(len(p)), sorted_indexes[:, -1]]
second_highest_p = p[np.arange(len(p)), sorted_indexes[:, -2]]

highest_p_var = var[np.arange(len(p)), sorted_indexes[:, -1]]
second_highest_p_var = var[np.arange(len(p)), sorted_indexes[:, -2]]

accept_classification = highest_p - highest_p_var > second_highest_p + second_highest_p_var 

remaining_incorrect_indices = np.argwhere((predictions != y_test) & (accept_classification == True))
missed_correct_indices = np.argwhere((predictions == y_test) & (accept_classification == False))

correct_accept = len(np.argwhere((predictions == y_test) & (accept_classification == True)))
correct_reject = len(np.argwhere((predictions == y_test) & (accept_classification == False)))
incorrect_accept = len(np.argwhere((predictions != y_test) & (accept_classification == True)))
incorrect_reject = len(np.argwhere((predictions != y_test) & (accept_classification == False)))

print("correct_accept: {}".format(correct_accept))
print("correct_reject:{}".format(correct_reject))
print("incorrect_accept: {}".format(incorrect_accept))
print("incorrect_reject:{}".format(incorrect_reject))

print("Rejecting classification for {} points".format(len(p) - len(np.argwhere(accept_classification == True))))

print("Remaining number of wrong classifications: {}".format(len(remaining_incorrect_indices)))

print("Number of correct classifications rejected: {}".format(len(missed_correct_indices)))

misclassification_costs = [0.5, 1, 2, 10, 50]
for misclassification_cost in misclassification_costs:
    print("Cost of misclassification: {}".format(misclassification_cost))
    A = (len(np.argwhere(predictions == y_test)) - misclassification_cost * len(np.argwhere(predictions != y_test))) / len(predictions)
    A_with_reject = (correct_accept - misclassification_cost * incorrect_accept) / len(y_test)
    print("New acc without rejection: {}".format(A))
    print("New acc with rejection: {}".format(A_with_reject))
