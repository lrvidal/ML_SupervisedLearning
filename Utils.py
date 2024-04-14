import random as rd
import numpy as np

def trainTestSplit(dataset, trainSize=0.8):
    trainLen = int(len(dataset) * trainSize)

    trainSet = []
    testSet = []

    trainIndices = rd.sample(range(len(dataset)), trainLen)

    for i in range(len(dataset)):
        if i in trainIndices:
            trainSet.append(dataset[i])
        else:
            testSet.append(dataset[i])
    
    return trainSet, testSet

def normalize(X, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
    # Add a small constant to the denominator to avoid division by zero
    return (X - min_val) / (max_val - min_val + 1e-7)

def meanSquaredError(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)