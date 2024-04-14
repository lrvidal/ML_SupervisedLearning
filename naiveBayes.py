from FileParser import FileParser
import Utils as utils
import numpy as np

class naiveBayes:
    def __init__(self, file, eta, iterations):
        self.data = FileParser(file).data
        self.eta = eta
        self.iteration = iterations

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X = utils.normalize(X)

#for some fucking reason OG file wouldnt work, made copy
model = naiveBayes(file = 'ML_SupervisedLearning\emailsCopy.csv', eta = 0.2, iterations = 1000)

#Calcuate intial probabilities from dataset
numHam =0
numSpam =0
numTot =0
trainData, testData = utils.trainTestSplit(model.data, 0.8)

for item in trainData:
    if item[-1] == 0:
        numHam += 1
    if item[-1] == 1:
        numSpam += 1

probHam = (numHam / len(trainData))*100
probSpam = (numSpam / len(trainData))*100

print("\nReal:{}  Spam:{} ProbHam: {:0.2f}% ProbSpam: {:0.2f}%".format(numHam,numSpam,probHam,probSpam))


#model.fit([item[:-1] for item in trainData], [item[-1] for item in trainData])
