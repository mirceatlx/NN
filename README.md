# NN
neural networks:)

Digit recognition

Validation and Test sets : 10,000 images each
Train set : 50,000 images

NeuralNet:

A bad implementation of an ANN inspired by Michael Nielsen's variant. No Tensorflow or Pytorch, just plain matrix 
multplication (numpy) and partial derivatives.


The dataset used is MNIST, images being resized to 28x28 pixels, before training and evaluation. 

nn.py contains a model that uses the sigmoid function as activation for neurons and using the mean squared error function.
So the input layer has 784 neurons each being represented as a pixel, followed by a single hidden layer and an 
output layer with 10 neurons, each representing a digit. 

nn_optim.py is a better and more optimized version of the neural network, as we intrduce the Cross Entropy cost function
which prevent neuron saturation, compared to the mean squared error. 

Another improvement is in the initialization of weights and biases in a random fashion which gives way better results as 
gradient descent does not become stuck in a local optima (hopefully). 


In the tfvariant folder, there is a Tensorflow equivalent implementation of the NeuralNet, using the GPU when training
(instead of the CPU) for max performance.

Required libraries:

    - numpy 
    - tensorflow and tensorflow_datasets (only for tfvariant)
    - gzip 
    - pickle


Run the models:

    python execute.py

