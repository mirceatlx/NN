import gzip
import numpy as np
import pickle
import os

def preload():
    """
    Uncompress the data from the MNIST dataset.
    Validation and Test contain 10,000 images each.
    Training contains 50,000 images. The data is represented as tuples.
    """
    os.chdir('..')
    f = gzip.open("data/mnist.pkl.gz", 'rb')
    train, vald, test = pickle.load(f, encoding = 'latin1')
    f.close()
    return train, vald, test

def load():
    train , vali, test = preload()

    train_data = [np.reshape(x , (784,1)) for x in train[0]]
    test_data = [np.reshape(x , (784,1)) for x in test[0]]
    vali_data = [np.reshape(x , (784,1)) for x in vali[0]]

    train_rez = [transform(y) for y in train[1]]
    test_rez = [transform(y) for y in test[1]]
    vali_rez = [transform(y) for y in vali[1]]

    xtrain = list(zip(train_data, train_rez))
    xtest = list(zip(test_data, test_rez))
    xvali = list(zip(vali_data, vali_rez))

    #print(np.shape(train_data[0]), np.shape(train_rez[0]))

    return xtrain, xvali, xtest


def transform(index):
    """
    Transform the output of the dataset into a 10 dimension vector.
    """
    v = np.zeros((10,1))
    v[index] = 1
    return v


t, v, tt = load()

print(t[0][0])

