from DecisionTrees import DecisionTrees
from FileParser import FileParser
import Utils as utils
import matplotlib.pyplot as plt
import time

data = FileParser("emails.csv").data
m = DecisionTrees()
trainSet, testSet = utils.trainTestSplit(data, 0.7)

start = time.time()
m.fit(trainSet)

rightPredictions = 0
testVals = [item[:-1] for item in testSet] 
testLabels = [item[-1] for item in testSet]  
pred = m.predict(testVals)


for i in range(len(testLabels)):
    if pred[i] == testLabels[i]:
        rightPredictions += 1

print(f'Accuracy of Decision Tree: {rightPredictions / len(testSet) * 100:.2f}%')

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
plt.title('Confusion Matrix For Decision Tree Classifier')
plt.show()