from __future__ import division  # floating point division
import numpy as np
import utilities as utils
from math import log
from sklearn.preprocessing import normalize

#  random seed initiallize

np.random.seed(2488)

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': False}
        self.class0Mean = 0
        self.class1Mean = 0
        self.class0Std = 0
        self.class1Std = 0
        self.reset(parameters)
        self.lastCol = 8 if self.params['usecolumnones'] == False else 9

            
    def reset(self, parameters):
       self.resetparams(parameters)

        # TODO: set up required variables for learning

    def learn(self, Xtrain, ytrain):
        '''
        this function will learn the Gaussian parameters for each features given the class
        :param Xtrain: Train feature data
        :param ytrain: Train class data
        :return: None
        '''
        print self.getparams()
        class0List = []
        class1List= []
        for index in range(ytrain.size):
            if ytrain[index,] == np.float64(1):
                class1List.append(index)
            else:
                class0List.append(index)



        self.class0Mean = np.mean(Xtrain[class0List,0:self.lastCol],axis=0)
        self.class1Mean = np.mean(Xtrain[class1List,0:self.lastCol],axis=0)
        self.class0Std = np.std(Xtrain[class0List,0:self.lastCol],axis=0)
        self.class1Std = np.std(Xtrain[class1List,0:self.lastCol],axis=0)


    def predict(self, Xtest):
        testSize = Xtest.shape[0]
        ytest = np.zeros(testSize)
        for index in range(testSize):
            testData = Xtest[index,0:self.lastCol]
            prob0 = 0
            prob1 = 0
            for ft in range(self.lastCol):
                try:
                    prob0 += log(utils.calculateprob(Xtest[index,ft],self.class0Mean[ft, ],self.class0Std[ft, ]))
                    prob1 += log(utils.calculateprob(Xtest[index,ft],self.class1Mean[ft, ],self.class1Std[ft, ]))
                except ValueError:
                        if utils.calculateprob(Xtest[index,ft],self.class0Mean[ft, ],self.class0Std[ft, ]) == 0:
                            prob0 += 0
                        elif utils.calculateprob(Xtest[index,ft],self.class1Mean[ft, ],self.class1Std[ft, ])== 0:
                            prob1 += 0
                ytest[index] = np.float64(0) if prob0 > prob1 else np.float(1)
        return ytest
            
class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.epochs = 500
        self.stepSize = 0.9
        self.reset(parameters)


    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def learn(self, Xtrain, ytrain):
        initial = np.random.rand(Xtrain.shape[1],1)
        self.weights = initial/utils.l2(initial)
        epoch = 1
        count = 1
        while epoch < self.epochs:
            if epoch%50== 0:
                count += 1
                self.stepSize= pow(self.stepSize,count)
            gradient = self.calculate_grad(Xtrain,ytrain)
            if self.params['regularizer'] is 'l1':
                self.weights = np.add(self.weights, np.add(-1*np.sign(self.weights)*self.params['regwgt'] , gradient))
            elif self.params['regularizer'] is 'l2':
                self.weights =np.add(self.weights, np.add(-1*self.weights*self.params['regwgt'] , gradient))
            else:
                self.weights += gradient

            epoch += 1

    def compute_cost(self,Xtrain,ytrain):
        '''
        Compute the cost  for the logistic regression
        :param Xtrain: Training data features
        :param ytrain: Training data class
        :return: cost
        '''
        dataSize = Xtrain.shape[0]
        vecOnes = np.ones(dataSize)
        hvalue = utils.sigmoid(Xtrain.dot(self.weights))

        cost = (-1/dataSize)*np.add(ytrain.dot(np.log(hvalue)) , np.subtract(vecOnes,ytrain).dot(np.log(np.subtract(vecOnes,hvalue))))
        cost = np.sum(cost,axis=0)
        return cost


    def calculate_grad(self,Xtrain,ytrain):
        '''
        Calculate the gradient after each epoch and return it
        :param Xtrain:
        :param ytrain:
        :return:
        '''
        gradVector = np.zeros(Xtrain.shape[1])
        predict = utils.sigmoid(Xtrain.dot(self.weights))
        delta = np.subtract(ytrain.reshape(ytrain.size,1),predict)
        sumGrad = Xtrain.T.dot(delta)
        gradVector = (1/Xtrain.shape[0])*sumGrad
        return gradVector

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest = utils.sigmoid(ytest)
        ytest[ytest > 0.5] = 1
        ytest[ytest < 0.5] = 0
        ytest = np.squeeze(ytest)
        return ytest
     
    # TODO: implement learn and predict functions


class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.9,
                        'epochs': 200}
        self.reset(parameters)
        self.tolerance = 10e-5
        self.hiddenNodes = self.params['nh']


    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.params['stepsize'] = 0.9
        self.wi = None
        self.wo = None

    # TODO: implement learn and predict functions


    def _evaluate(self, inputs):
        """
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.size != self.inputNodes:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')

        # hidden activations
        ah = self.transfer(np.dot(self.wi.T,inputs))
        # output activations.
        ao = self.transfer(np.add(np.dot(self.wo.T,ah),self.biasO))
        return (ah, ao)



    def _backward(self,hidden,output,trueClass,inputs):

        # For last layer  weight update

        delta2 = np.multiply(utils.dsigmoid(output),np.subtract(output,trueClass))
        update2 = delta2.T.dot(hidden.reshape(1,hidden.size))
        update2 = update2 * self.params['stepsize']


        # For first layer weight update
        delta1 =  self.wo.dot(delta2.T)
        delta1 = np.multiply(utils.dsigmoid(hidden).reshape(hidden.size,1),delta1)
        update1 = delta1.dot(inputs.reshape(1,inputs.size))
        update1 = update1 * self.params['stepsize']



        #update both the weights
        self.wo = np.subtract(self.wo, update2.T)
        bias = np.sum(update2, axis=1)
        self.biasO = np.subtract(self.biasO, bias)
        self.wi = np.subtract(self.wi,update1.T)




    def learn(self, Xtrain, ytrain):
        self.inputNodes = Xtrain.shape[1]
        outputNodes = 2
        self.wi = np.random.normal(0, 1, self.inputNodes * self.hiddenNodes).reshape(self.inputNodes, self.hiddenNodes)
        self.wo = np.random.normal(0, 1, outputNodes * self.hiddenNodes).reshape(self.hiddenNodes,outputNodes)
        self.biasO = np.ones((1,outputNodes))
        epoch = 0
        count = 1
        while epoch < self.params['epochs']:
            shuffleList = np.arange(Xtrain.shape[0])
            if epoch%5 == 0:
                count += 1
                self.params['stepsize']= pow(self.params['stepsize'],count)
            np.random.shuffle(shuffleList)
            for index in shuffleList:
                data = Xtrain[index,:]
                actualClass = np.zeros((1,outputNodes))
                actualClass[:,int(ytrain[index])] = np.float64(1)
                (ah,ao) = self._evaluate(data)
                self._backward(ah,ao,actualClass,data)
            epoch += 1


    def predict(self, Xtest):
        ylayer1 = np.dot(Xtest, self.wi)
        hiddenOut = utils.sigmoid(ylayer1)
        ylayer2 = hiddenOut.dot(self.wo)
        ylayer2 = np.add(ylayer2,self.biasO)
        predicted = utils.sigmoid(ylayer2)
        ytest = np.zeros(Xtest.shape[0])
        for index in range(Xtest.shape[0]):
            if predicted[index,0] > predicted[index,1]:
                ytest[index]= np.float64(0)
            else:
                ytest[index] = np.float64(1)
        return ytest

class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.params = {'Lambda1': 0.002, 'Lambda2': 0.005}
        self.epochs = 500
        self.stepSize = 0.9
        self.reset(parameters)


    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        
    # TODO: implement learn and predict functions

    def learn(self, Xtrain, ytrain):
        initial = np.random.rand(Xtrain.shape[1],1)
        self.weights = initial/utils.l2(initial)
        epoch = 1
        count = 1
        while epoch < self.epochs:
            if epoch%50== 0:
                count += 1
                self.stepSize= pow(self.stepSize,count)
            gradient = self.calculate_grad(Xtrain,ytrain)

            regularizeTerm = -1*np.add(np.sign(self.weights)*self.params['Lambda1'],self.weights*self.params['Lambda2'])

            self.weights = np.add(self.weights, np.add(regularizeTerm , gradient))


            epoch += 1
           
    def calculate_grad(self,Xtrain,ytrain):
        '''
        Calculate the gradient after each epoch and return it
        :param Xtrain:
        :param ytrain:
        :return:
        '''
        gradVector = np.zeros(Xtrain.shape[1])
        predict = utils.sigmoid(Xtrain.dot(self.weights))
        delta = np.subtract(ytrain.reshape(ytrain.size,1),predict)
        sumGrad = Xtrain.T.dot(delta)
        gradVector = (1/Xtrain.shape[0])*sumGrad
        return gradVector

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest = utils.sigmoid(ytest)
        ytest[ytest > 0.5] = 1
        ytest[ytest < 0.5] = 0
        ytest = np.squeeze(ytest)
        return ytest

class NewLogitReg(Classifier):
    def __init__(self, parameters={}):
        self.params = {}
        self.epochs = 500
        self.stepSize = 0.01
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    # TODO: implement learn and predict functions

    def learn(self, Xtrain, ytrain):
        initial = np.random.rand(Xtrain.shape[1], 1)
        self.weights = initial / utils.l2(initial)
        epoch = 1
        count = 1
        while epoch < self.epochs:
            if epoch % 50 == 0:
                count += 1
                self.stepSize = pow(self.stepSize, count)
            gradient = self.calculate_grad(Xtrain, ytrain)
            self.weights += gradient
            # print "cost: ",self.compute_cost(Xtrain,ytrain)
            epoch += 1

    def calculate_grad(self, Xtrain, ytrain):
        '''
        Calculate the gradient after each epoch and return it
        :param Xtrain:
        :param ytrain:
        :return:
        '''

        predict = Xtrain.dot(self.weights)
        sqpredict = np.power(predict,2)
        prob = 1/2 * (1 + predict / np.power((1+sqpredict),0.5))
        delta = 1/(2 * np.power((1 + sqpredict),1.5))
        ytrain = ytrain.reshape(ytrain.size,1)
        gradient = (ytrain / prob) * delta - (1-ytrain)/(1-prob) * delta
        gradient = Xtrain.T.dot(gradient)
        gradient = gradient/ytrain.shape[0]
        return gradient

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest = utils.sigmoid(ytest)
        ytest[ytest > 0.5] = 1
        ytest[ytest < 0.5] = 0
        ytest = np.squeeze(ytest)
        return ytest

    def compute_cost(self,Xtrain,ytrain):
        '''
        Compute the cost  for the logistic regression
        :param Xtrain: Training data features
        :param ytrain: Training data class
        :return: cost
        '''
        dataSize = Xtrain.shape[0]
        vecOnes = np.ones(dataSize)
        hvalue = utils.sigmoid(Xtrain.dot(self.weights))

        cost = (-1/dataSize)*np.add(ytrain.dot(np.log(hvalue)) , np.subtract(vecOnes,ytrain).dot(np.log(np.subtract(vecOnes,hvalue))))
        cost = np.sum(cost,axis=0)
        return cost