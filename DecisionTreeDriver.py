from decisionTrees import DecisionTrees
from FileParser import FileParser
import Utils as utils

data = FileParser("emails.csv").data
m = DecisionTrees()
trainSet, testSet = utils.trainTestSplit(data, 0.7)
m.fit(trainSet)

rightPredictions = 0
testVals = [item[:-1] for item in testSet] 
testLabels = [item[-1] for item in testSet]  
pred = m.predict(testVals)


for i in range(len(testLabels)):
    if pred[i] == testLabels[i]:
        rightPredictions += 1

print(f'Accuracy of Decision Tree: {rightPredictions / len(testSet) * 100:.2f}%')