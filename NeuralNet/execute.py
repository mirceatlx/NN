from loader import *
from nn import *
from nn_optim import *

train , vali, test = load()

net = NeuralNOptim([784, 100, 10], cost = CrossEntropyCost)
net.SGD(train, 30, 10, 0.5, 5.0 ,vali, True, True, True, True)