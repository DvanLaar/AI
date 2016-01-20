# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        #Code
        resultingWeights = []
        startingWeights = self.weights.copy()
        for C in Cgrid:
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, ", with C-value ", C
                for i in range(len(trainingData)):
                    classified = self.classify([trainingData[i]])[0] #guess the i-th datum
                    if classified == trainingLabels[i]: #If guessed correctly, move on
                        continue
                    tau = min(C, ((self.weights[classified]-self.weights[trainingLabels[i]])*trainingData[i]+1.0)/((trainingData[i] + trainingData[i])*trainingData[i]))
                    #Adjust the weights
                    self.weights[classified] -= self.counterMul(tau,trainingData[i])
                    self.weights[trainingLabels[i]] += self.counterMul(tau,trainingData[i])
            resultingWeights.append(self.weights.copy())
            self.weights = startingWeights.copy()
        scoreOfCValues = util.Counter()
        for C in range(len(Cgrid)):
            correct = 0
            self.weights = resultingWeights[C].copy()
            classifiedStuff = self.classify(validationData) #Guess the validationData using the resultingWeights for this C value
            #Test if the guesses are correct
            for i in range(len(validationData)):
                if classifiedStuff[i] == validationLabels[i]:
                    correct += 1
            scoreOfCValues[C] = correct
        #Find the best C-value    
        bestCIndex = scoreOfCValues.argMax()
        self.weights = resultingWeights[bestCIndex]
    
    #Help function to multiply a counter y by an integer x.
    def counterMul(self, x, y):
        result = util.Counter()
        keys = y.keys()
        for key in keys:
            result[key] = x * y[key]
        return result
            
        #Code
        "*** END MY CODE ***"

        
    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        #Code
        tempWeights = self.weights[label].copy()
        for i in range(100):
            best = tempWeights.argMax()
            featuresWeights.append(best)
            tempWeights.pop(best)
        #Code
        "*** END MY CODE ***"

        return featuresWeights