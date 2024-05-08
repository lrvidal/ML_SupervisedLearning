from FileParser import FileParser
import Utils as utils
import numpy as np
import matplotlib.pyplot as plt

alpha =1
class naiveBayes:
    def __init__(self, file, eta, iterations):
        self.data = FileParser(file).data
        self.eta = eta
        self.iteration = iterations

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X = utils.normalize(X)

print("Naive Bayes Time :3\n")

#for some fucking reason OG file wouldnt work, made copy
model = naiveBayes(file = 'ML_SupervisedLearning\emailsCopy.csv', eta = 0.2, iterations = 1000)
trainData, testData = utils.trainTestSplit(model.data, 0.8)

#Calcuate intial probabilities from dataset
numHam =0
numSpam =0
numTot =0

for item in trainData:
    if item[-1] == 0:
        numHam += 1
    if item[-1] == 1:
        numSpam += 1

#the,too,ect
probSpamList = [0] * len(trainData)
probHamList = [0] * len(trainData)

testSpamList = np.zeros((4137, 4137))

#Laplace Smoothing
for item in trainData:
    
    if item[-1] == 0:
        for j in range(len(item)):
            #only saves last item TODO
            probHamList[j] = (item[j]+alpha)/((len(item))*alpha)

    if item[-1] == 1:
         for j in range(len(item)):
            #only saves last item TODO
            probSpamList[j] = (item[j]+alpha)/((len(item))*alpha)
            
probHam = (numHam / len(trainData))*100
probSpam = (numSpam / len(trainData))*100
print(testSpamList)

print("\nAnalyzing provided Dataset...")
print("\nReal:{}  Spam:{} ProbHam: {:0.2f}% ProbSpam: {:0.2f}%".format(numHam,numSpam,probHam,probSpam))

#################
rightPredictions= 0
#for item in testData:
    #pred = model.predict(item[:-1])
    #if pred == item[-1]:
    #    rightPredictions += 1

print(f'Accuracy: {rightPredictions / len(testData) * 100:.2f}%')
#print(f'MSE: {utils.meanSquaredError(y_true=[item[-1] for item in testData], y_pred=[model.predict(item[:-1]) for item in testData])}')


######################
#test1

#fuck it Pi graph
values = [probSpam, probHam] 
labels = ['"Spam"', '"Ham"']

plt.pie(values, labels=labels, startangle=90, autopct='%1.1f%%')

plt.title('Initial Dataset Assessment')
plt.axis('equal')
plt.show()