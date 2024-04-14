import random as rd

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

def meanSquaredError(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)