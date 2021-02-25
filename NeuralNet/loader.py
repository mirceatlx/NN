import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def load():
    """
    Prepare the data to be fed to the Neural Network.
    Preprocessing.
    """
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    y_train = train['label'].reset_index(drop = True)
    train = train.drop(['label'], axis = 1)
    train = train.to_numpy()
    test = test.to_numpy()

    y_train_transform = [transform_label(y) for y in y_train]
    train_transform = [transform_tt(x) for x in train]
    test_transform = [transform_tt(x) for x in test]

    return train_transform, y_train_transform, test_transform


def transform_tt(X):
    """
    Transform the training and test datasets to be suitable to our NN.
    Return a numpy array of (784,1) shape. (One training example)
    """
    X = np.reshape(X, (784,1))
    return X

def transform_label(label):
    """
    Return a 10-dimensional unit vector where the j-th position is 1 
    and zeroes everywhere else.
    """

    v = np.zeros((10,1))
    v[label] = 1
    return v

load()

