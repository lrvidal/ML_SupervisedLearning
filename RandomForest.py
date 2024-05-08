import random
from decisionTrees import DecisionTrees
from FileParser import FileParser
import Utils as utils


class RandomForest:
    def __init__(self, number_trees, filename):
        self.data = FileParser(filename).data
        self.number_trees = number_trees
        self.forest = []

    def fit(self, data):
        for _ in range(self.number_trees):
            # Create a bootstrap sample of the data
            sample = [random.choice(data) for _ in range(len(data))]

            # Create a decision tree and fit it to the bootstrap sample
            tree = DecisionTrees()
            tree.fit(sample)

            # Add the tree to the forest
            self.forest.append(tree)

    def predict(self, input_data):
        # Make a prediction with each tree
        predictions = [tree.predict(input_data) for tree in self.forest]

        # Take the majority vote of the predictions
        majority_vote = [max(set(preds), key=preds.count) for preds in zip(*predictions)]

        return majority_vote
    
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

