from FileParser import FileParser
import Utils as utils
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, filename, eta=0.2, iterations=1000):
        self.data = FileParser(filename).data
        self.eta = eta
        self.iteration = iterations
        self.weights = None
        self.min_val = None
        self.max_val = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Normalize X
        X = utils.normalize(X) #

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
        X = utils.normalize(X, self.min_val, self.max_val)

        X = np.insert(X, 0, 1)
        return self.sigmoid(np.dot(X, self.weights)) >= 0.5
    

m = LogisticRegression(filename='ML_SupervisedLearning\emailsCopy.csv', eta=0.3, iterations=1000)
trainSet, testSet = utils.trainTestSplit(m.data, 0.7)
m.fit([item[:-1] for item in trainSet], [item[-1] for item in trainSet])

rightPredictions = 0
for item in testSet:
    pred = m.predict(item[:-1])
    if pred == item[-1]:
        rightPredictions += 1

print(f'Accuracy of Logistic Regression: {rightPredictions / len(testSet) * 100:.2f}%')

values = [rightPredictions / len(testSet) * 100, 100-rightPredictions / len(testSet) * 100] 
labels = ['"Spam Predictions"', '"Ham Predictions"']

plt.pie(values, labels=labels, startangle=90, autopct='%1.1f%%')

plt.title('Logistic Regression Accuracy')
plt.axis('equal')
plt.show()
