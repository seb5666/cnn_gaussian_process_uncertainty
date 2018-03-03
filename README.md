# CNN uncertainty with Gaussian processes

Convolutional neural networks (CNNs) achieve state-of-the-art performance on image classification tasks, but provide no measure of confidence in their predictions. I implement and analyse on way to obtain confidence or uncertainty bounds in their predictions: After training a CNN, I use the outputs from the top level feature layer before the softmax classifier to train a Gaussian Process classifier on the MNIST data. 

## Results
All results are reported in my written report [link](report/report.pdf).

## Code
The entry point to train and evaluate this approach can be found in [GP_uncertainty_on_mnist.py](GP_uncertainty_on_mnist.py). The other contain code for visualization and for a more in depth analysis of the results.
