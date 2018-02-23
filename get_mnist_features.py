import keras
from keras.datasets import mnist
from keras import backend as K
import numpy as np

def get_gp_mnist_data(model_file, X):
    """
        Computes the activations of the last layer of a CNN trained on MNIST for the training
        and test data for the MNIST dataset
    """

    model = keras.models.load_model(model_file)
    
    return _get_ith_layer_output(model, X, -2)

def _get_ith_layer_output(model, X, i, mode='test', batch_size=1024):
    ''' see https://keras.io/getting-started/faq/#keras-faq-frequently-asked-keras-questions'''
#    get_ith_layer = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()],[model.layers[i].output])
#    layer_output = get_ith_layer([X, 0 if mode=='test' else 1])[0]
#    return layer_output

    get_ith_layer = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()],[model.layers[i].output])
    layer_output = []
    batch_start = 0
    while batch_start < len(X):
        outputs = get_ith_layer([X[batch_start:batch_start+batch_size], 0 if mode=='test' else 2])[0]
        layer_output.append(outputs)
        batch_start += batch_size
    layer_output = np.concatenate(layer_output)
    return layer_output
