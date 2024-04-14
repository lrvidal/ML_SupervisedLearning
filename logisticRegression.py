from FileParser import FileParser
import Utils as utils
import numpy as np

class LogisticRegression:
    def __init__(self, filename, eta=0.2, iterations=1000):
        self.data = FileParser(filename).data
        self.eta = eta
        self.iteration = iterations
        self.max_val = None
        self.min_val = None

    def normalize(self, X, min_val=None, max_val=None):
        if min_val is None or max_val is None:
            min_val = np.min(X, axis=0)
            max_val = np.max(X, axis=0)
            self.min_val = min_val
            self.max_val = max_val
        # Add a small constant to the denominator to avoid division by zero
        return (X - min_val) / (max_val - min_val + 1e-7)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Normalize X
        X = self.normalize(X)

        self.weights = np.zeros(X.shape[1] + 1)
        X = np.insert(X, 0, 1, axis=1)

        for i in range(self.iteration):
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weights -= self.eta * gradient

    def predict(self, X):
        X = np.array(X)

        # Normalize X
        X = self.normalize(X, self.min_val, self.max_val)

        X = np.insert(X, 0, 1)
        return self.sigmoid(np.dot(X, self.weights)) >= 0.5
    

m = LogisticRegression(filename='emails.csv', eta=0.3, iterations=1000)
trainSet, testSet = utils.trainTestSplit(m.data, 0.7)
m.fit([item[:-1] for item in trainSet], [item[-1] for item in trainSet])

rightPredictions = 0
for item in testSet:
    pred = m.predict(item[:-1])
    if pred == item[-1]:
        rightPredictions += 1

print(f'Accuracy: {rightPredictions / len(testSet) * 100:.2f}%')
print(f'MSE: {utils.meanSquaredError(y_true=[item[-1] for item in testSet], y_pred=[m.predict(item[:-1]) for item in testSet])}')
