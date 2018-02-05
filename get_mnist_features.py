import keras
from keras.datasets import mnist
from keras import backend as K
import numpy as np

def get_gp_mnist_data(model_file):
    """
        Computes the activations of the last layer of a CNN trained on MNIST for the training
        and test data for the MNIST dataset
    """
    model = keras.models.load_model(model_file)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    num_classes = 10
    y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
    y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    #Handle Theano and Tensorflow data format
    img_rows, img_cols = 28,28

    print("Image data format: {}".format(K.image_data_format()))

    if K.image_data_format() == 'channel_first':
        X_train = x_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = x_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)


    #evaluate CNN
    score = model.evaluate(X_test, y_test_one_hot, verbose=0)
    print("Test loss: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1]))

    last_layer_training_outputs = _get_ith_layer_output(model, X_train, -2)
    last_layer_test_outputs = _get_ith_layer_output(model, X_test, -2)

    return (X_train, last_layer_training_outputs, y_train), (X_test, last_layer_test_outputs, y_test)

def _get_ith_layer_output(model, X, i, mode='test'):
    ''' see https://keras.io/getting-started/faq/#keras-faq-frequently-asked-keras-questions'''
    get_ith_layer = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()],[model.layers[i].output])
    layer_output = get_ith_layer([X, 0 if mode=='test' else 1])[0]
    return layer_output
