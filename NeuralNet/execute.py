from loader import *
from nn import *
from nn_optim import *

train , vali, test = load() # using the MNIST dataset

print("Option 1: Simple implemenation of a Neural Network")
print("Option 2: Optimized implementation of a Neural Network")

option = int(input())


while option != 0:
  if option == 1:
    net = NeuralN([784, 100, 10])
    net.SGD(train, 50, 10, 1, test)
  if option == 2:
    net = NeuralNOptim([784, 100, 10], cost = CrossEntropyCost)
    net.SGD(train, 50, 10, 0.5, 5.0 ,vali, True, True, True, True)
  
  print("################## COMPLETED ###################")

  option =  int(input("Choose again: "))
