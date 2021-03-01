import numpy as np
import random
from loader import *


class NeuralN:
    
    def __init__(self, layers):
        self.numlayers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]] # the input layer does not have bias
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def structure(self):
        print(f"The neural net has {self.numlayers} layers of neurons")
    
    def forward(self, a):
        """
        Returns the output of the network when a is the input.
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch, eta, test_data = None):
        """
        Training of the network using stochastic gradient descent.
        """
        if test_data:
            ntest = len(list(test_data))

        ntrain = len(list(training_data))
        for j in range(epochs):
            random.shuffle(list(training_data))
            mini_batches = [
                training_data[k : k + mini_batch]
                for k in range(0, ntrain, mini_batch)
            ]
            for batch in mini_batches:
                self.update(batch, eta) # where the magic happens
            if test_data:
                print ("Epoch {0} : {1} / {2} Accuracy : {3}%".format(j, self.evaluate(test_data), ntest, (self.evaluate(test_data) / ntest * 100.0)))
            else:
                print ("Epoch {0} completed".format(j))

    def update(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]
        self.weights = [
            W - (eta / len(mini_batch)) * nw
            for W, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            B - (eta / len(mini_batch)) * nb
            for B, nb in zip(self.biases, nabla_b)
        ]
            
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forward propagation
        a = x
        activs = [x]
        zactivs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zactivs.append(z)
            a = sigmoid(z)
            activs.append(a)

        # backward propagation
        delta = self.cost_derivative(activs[-1], y) * \
            sigmoid_prime(zactivs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activs[-2].transpose())

        for layer in range(2, self.numlayers):
            z = zactivs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activs[-layer-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        results = [(np.argmax(self.forward(x)), y) for x, y in test_data]
        return sum( int(np.array_equal(transform(x),y)) for x, y in results)

    def cost_derivative(self, output, y):
        return output - y


def sigmoid(z):
        """
        Activation function of our network.
        """
        return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


