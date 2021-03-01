import random
import numpy as np
from nn import *

class CrossEntropyCost():
    """
    Representation of Cross-Entropy Cost Function.
    Prevents slow learning.
    """

    def fuct(a, y):
        """
        The function measures how well an output activation a, matches the desired output y.
        """
        # np.nan_to_sum assures the function functions correctly near 0
        return np.sum(np.nan_to_num(- y * np.log(a) - (1 - y) * np.log(1 - a)))

    def delta(z, a, y):
        return a - y

class QuadraticCost():
    """
    Representation of the classic Quadratic Cost Function.
    It learns really slowly when the activation is close to 0 or 1.
    """

    def fuct(a, y):
        """
        The function measures how well an output activation a, matches the desired output y.
        """
        return 0.5 * np.linalg.norm(a - y)**2

    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)

class NeuralNOptim():

    def __init__(self, layers, cost=CrossEntropyCost):
        self.numlayers = len(layers)
        self.layers = layers
        self.default_weight_init()
        self.cost = cost

    def default_weight_init(self):
        """
        A good way to init weights in order to prevent the saturation of neurons.
        Not good in production.
        """

        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                                        for x, y in zip(self.layers[:-1], self.layers[1:])]

    def vanilla_weight_init(self):
        """
        The method to init weights and biases used in the first implementation.
        """
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(
            self.layers[:-1], self.layers[1:])]

    def forward(self, a):
        """
        Returns the output of the network when a is the input.
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, train_data, epochs, mini_batch, eta, lmbda=0.0, eval_data=None, monitor_eval_cost=False, monitor_eval_acc=False, monitor_train_cost=False, monitor_train_acc=False):
        """
        We can monitor the cost and accuracy on either the eval data or the training data, by setting
        appropiate flags. The method returns a tuple containing four lists.
        """
        if eval_data:
            n_data = len(eval_data)  # either validation or test data
        n = len(train_data)
        eval_cost, eval_acc = [], []
        train_cost, train_acc = [], []
        for j in range(epochs):
            random.shuffle(train_data)
            mini_batches = [
                train_data[k:k + mini_batch]
                for k in range(0, n, mini_batch)
            ]
            for batch in mini_batches:
                self.update(batch, eta, lmbda, n)

            if monitor_train_cost:
                cost = self.total_cost(train_data, lmbda)
                train_cost.append(cost)
                print("Cost on training data: {}".format(cost))

            if monitor_train_acc:
                acc = self.accuracy(train_data, convert=True)
                train_acc.append(acc)
                print("Accuracy on training data: {} / {}".format(acc, n))

            if monitor_eval_cost:
                cost = self.total_cost(eval_data, lmbda, convert=True)
                eval_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))

            if monitor_eval_acc:
                acc = self.accuracy(eval_data)
                eval_acc.append(acc)
                print("Accuracy on evaluation data: {} / {}".format(acc, n_data))

            print("***************** Epoch {0} *****************".format(j))

        return eval_cost, eval_acc, train_cost, train_acc

    def update(self, mini_batch, eta, lmbda, n):

        nabla_b = [np.zeros(y.shape) for y in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]

        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w) 
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
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
        delta = (self.cost).delta(zactivs[-1], activs[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activs[-2].transpose())

        for layer in range(2, self.numlayers):
            z = zactivs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activs[-layer-1].transpose())
        return nabla_b, nabla_w            

    def accuracy(self, data, convert = False):
        """
        The flag 'convert' should be set to False if the data set is validation or test data and True
        if we deal with the training data.
        """

        if convert:
            results = [(np.argmax(self.forward(x)), np.argmax(y)) for x, y in data]
            return sum( int(x == y) for x, y in results)

        else:
            results = [(np.argmax(self.forward(x)), y) for x, y in data]
            return sum( int(np.array_equal(transform(x),y)) for x, y in results)
    
        #print(transform(np.argmax(self.forward(data[0][0]))))
        

    def total_cost(self, data, lmbda, convert = False):
        """
        Returns the total cost for the data set. The flag tutorial can be seen in 'accuracy'. (but reversed)
        """             
        cost = 0.0
        for x, y in data:
            a = self.forward(x)
            #if convert:
                #y = transform(y)
            cost += self.cost.fuct(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """
        Optional function to use th Neural Net.
        """            
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()






# Miscellaneous functions
def transform(index):
    v = np.zeros((10,1))
    #print(index.shape)
    v[index] = 1
    return v

def sigmoid(z):
    """
    Activation function of our network.
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))    

            
