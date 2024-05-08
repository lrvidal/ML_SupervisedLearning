import random
from DecisionTrees import DecisionTrees
from FileParser import FileParser
import Utils as utils
import matplotlib.pyplot as plt
import time


class RandomForest:
    def __init__(self, number_trees, filename):
        self.data = FileParser(filename).data
        self.number_trees = number_trees
        self.forest = []

    def fit(self, data):
        for _ in range(self.number_trees):
            sample = [random.choice(data) for _ in range(len(data))]

            tree = DecisionTrees()
            tree.fit(sample)

            self.forest.append(tree)

    def predict(self, input_data):
        predictions = [tree.predict(input_data) for tree in self.forest]

        majority_vote = [max(set(preds), key=preds.count) for preds in zip(*predictions)]

        return majority_vote
    

start = time.time()

m = RandomForest(filename='emails.csv', number_trees=10)
trainSet, testSet = utils.trainTestSplit(m.data, 0.7)
m.fit(trainSet)

rightPredictions = 0
testVals = [item[:-1] for item in testSet]
testLabels = [item[-1] for item in testSet]
pred = m.predict(testVals)

for i in range(len(testLabels)):
    if pred[i] == testLabels[i]:
        rightPredictions += 1

print(f'Accuracy of Random Forest: {rightPredictions / len(testSet) * 100:.2f}%')

confusion = utils.confusionMatrixGen(testLabels, pred)

print(f'True Positives: {confusion[0][0]}')
print(f'False Positives: {confusion[0][1]}')
print(f'True Negatives: {confusion[1][1]}')
print(f'False Negatives: {confusion[1][0]}')

end = time.time()
print(f'Time taken: {end - start} seconds')
# Plotting the confusion matrix as a pie chart
labels = ['True Positives', 'False Positives', 'True Negatives', 'False Negatives']
values = [confusion[0][0], confusion[0][1], confusion[1][1], confusion[1][0]]

plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title('Confusion Matrix For Random Forests Classifier')
plt.show()
