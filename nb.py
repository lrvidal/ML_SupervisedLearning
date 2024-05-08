from FileParser import FileParser
import Utils as utils
import numpy as np
import matplotlib.pyplot as plt


##NAIVE BAYES using the class, incorrect accuracy.##

class naiveBayes:
    def __init__(self, file, eta, iterations):
        self.data = FileParser(file).data
        self.eta = eta
        self.iteration = iterations

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Normalize X
        X = utils.normalize(X)

        # Calculate class probabilities
        self.class_probs = {}
        for label in np.unique(y):
            self.class_probs[label] = np.sum(y == label) / len(y)

        # Calculate feature probabilities
        self.feature_probs = {}
        for feature in range(X.shape[1]):
            self.feature_probs[feature] = {}
            for label in np.unique(y):
                X_label = X[y == label, feature]
                self.feature_probs[feature][label] = {
                    'mean': np.mean(X_label),
                    'std': np.std(X_label)
                }

    def predict(self, X):
        X = np.array(X)
        X = utils.normalize(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        predictions = []
        
        for i in range(X.shape[0]):
            probabilities = {}
            for label in self.class_probs:
                probabilities[label] = np.log(self.class_probs[label])
                for feature in range(X.shape[1]):
                    mean = self.feature_probs[feature][label]['mean']
                    std = self.feature_probs[feature][label]['std'] + 1e-9
                    x = X[i][feature]
                    exponent = -((x - mean) ** 2) / (2 * std ** 2)
                    probabilities[label] += np.log(1 / (np.sqrt(2 * np.pi) * std)) + exponent
            predicted_label = max(probabilities, key=probabilities.get)
            predictions.append(predicted_label)
        return predictions

model = naiveBayes(file = 'ML_SupervisedLearning\emailsCopy.csv', eta = 0.2, iterations = 1000)
trainData, testData = utils.trainTestSplit(model.data, 0.8)
model.fit([item[:-1] for item in trainData], [item[-1] for item in trainData])

#Calcuate intial probabilities from dataset
numHam, numSpam, numToT = 0, 0, 0
for item in trainData:
    if item[-1] == 0:
        numHam += 1
    if item[-1] == 1:
        numSpam += 1

total = numHam + numSpam
model.class_probs = {0: numHam / total, 1: numSpam / total}

########## TESTING ##########
rightPredictions = 0
for item in testData:
    pred = model.predict([item[:-1]])
    if pred[0] == item[-1]:
        rightPredictions += 1
#############################

print(f'Accuracy: {rightPredictions / len(testData) * 100:.2f}%')
#print(f'MSE: {utils.meanSquaredError(y_true=[item[-1] for item in testData], y_pred=[model.predict(item[:-1]) for item in testData])}')
