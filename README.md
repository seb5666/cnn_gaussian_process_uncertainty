# CNN uncertainty with Gaussian processes

## Main task
Convolutional neural networks (CNNs) achieve state-of-the-art performance on image classification tasks, but provide no measure of confidence in their predictions. Use the Keras MNIST example CNN available at [https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py). After training the model, use the outputs from the top level feature layer before the softmax classifier to train a Gaussian Process classifier on the MNIST data. Evaluate your classifier’s performance compared to the original CNN with softmax classifier. Plot the distribution of classification uncertainties for correctly and incorrectly classified samples. 

## Extension
Without re-training, use your model to classify the n-MNIST dataset [http://csc.lsu.edu/~saikat/n-mnist/](http://csc.lsu.edu/~saikat/n-mnist/), again plotting the classification uncertainties. Investigate how adversarial perturbations to the MNIST data affect classification accuracy and uncertainty (try the CleverHans toolbox for generating adversarial perturbations).
