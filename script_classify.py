from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import classalgorithms as algs
from sklearn import cross_validation as CV



def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

 
if __name__ == '__main__':
    trainsize = 8000
    testsize = 2000
    dataSize = trainsize+testsize
    numruns = 10

    classalgs = {'Random': algs.Classifier(),
                'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
                 'Linear Regression': algs.LinearRegressionClass(),
                 'Logistic Regression': algs.LogitReg(),
                 'L1 Logistic Regression': algs.LogitReg({'regularizer': 'l1'}),
                 'L2 Logistic Regression': algs.LogitReg({'regularizer': 'l2'}),
                 'Logistic Alternative': algs.LogitRegAlternative(),
                 'Problem2 Logistic': algs.NewLogitReg(),
                 'Neural Network': algs.NeuralNet({'epochs': 30})
                }  
    numalgs = len(classalgs)

    parameters = (
        {'regwgt': 0.0, 'nh': 4, 'Lambda1':0.01, 'Lambda2':0.09},
        {'regwgt': 0.01, 'nh': 8, 'Lambda1':0.02, 'Lambda2':0.05},
        {'regwgt': 0.05, 'nh': 16,'Lambda1':0.009, 'Lambda2':0.002},
        {'regwgt': 0.1, 'nh': 32,'Lambda1':0.002, 'Lambda2':0.005} )

    numparams = len(parameters)



    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    indexes = CV.ShuffleSplit(dataSize, n_iter=numruns, test_size=.25, random_state=2048)




    trainData, testData = dtl.load_susy_complete(trainsize, testsize)

    # trainData, testData = dtl.load_susy(trainsize, testsize)
    featuredata = np.concatenate((trainData[0], testData[0]), axis=0)
    classdata = np.concatenate((trainData[1], testData[1]), axis=0)

    r = 0
    for train_index,test_index in indexes:
    # for r in range(numruns):
    	# trainset, testset = dtl.load_susy(trainsize,testsize)

        trainft = featuredata[train_index]
        trainclass = classdata[train_index]
        testft = featuredata[test_index]
        testclass = classdata[test_index]

        trainset = (trainft,trainclass)
        testset = (testft,testclass)

    	# trainset, testset = dtl.load_susy_complete(trainsize,testsize)

        print('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r)

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.iteritems():
                # Reset learner for new parameters
                learner.reset(params)
    	    	print 'Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
    	    	# Train model
    	    	learner.learn(trainset[0], trainset[1])
    	    	# Test model
    	    	predictions = learner.predict(testset[0])
    	    	error = geterror(testset[1], predictions)
    	    	print 'Error for ' + learnername + ': ' + str(error)
                errors[learnername][p,r] = error
        r += 1

    for learnername, learner in classalgs.iteritems():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p


        # Extract best parameters
        learner.reset(parameters[bestparams])
        stderror = np.std(errors[learnername][bestparams,:])/ (0.75 * dataSize)
    	print 'Best parameters for ' + learnername + ': ' + str(learner.getparams())
    	print 'Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(1.96*np.std(errors[learnername][bestparams,:])/math.sqrt(numruns))
        print 'Standard error for ' + learnername + ': ' + str(stderror)
