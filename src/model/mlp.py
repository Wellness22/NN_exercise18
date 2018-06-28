
import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, hiddenLayerSizes=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], hiddenLayerSizes[0], 
                           None, inputActivation, False))

        # Hidden Layers
        for idx,sz in enumerate(hiddenLayerSizes[:-1]):
            self.layers.append(LogisticLayer(hiddenLayerSizes[idx], hiddenLayerSizes[idx+1], 
                           None, inputActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(hiddenLayerSizes[-1], 10, 
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        for layer in self.layers:
            print(layer.nIn)
            print(layer.nOut)
            print(layer.weights.shape)

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        out = inp
        for layer in self.layers:
            out = layer.forward(out)
            out = np.insert(out,0,1) # add bias coeff to front
        
    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self.loss.calculateDerivative(target, self.layers[-1].outp)
    
    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(learningRate)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))
                print(str(self.validationSet.label[:30]) + "... labels")
                print(str(np.array(self.evaluate(self.validationSet)[:30])) + "... outputs")

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")



    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        for img, label in zip(self.trainingSet.input,self.trainingSet.label):

            # Do a forward pass to calculate the output and the error, saves output at every neuron
            self._feed_forward(img)
            # build target output: 10 output neurons, one is labeled with "1" depending on the labels
            labelArr = np.zeros(10)
            labelArr[label] = 1.0

            # compute output error
            deltas = self._compute_error(labelArr)
            # initial weights just output (as matrix for arithmetical reasons)
            wgts = np.identity(self.layers[-1].nOut)

            # backpropagate through layers, calculate new delta based on delta and weights of previous layer, dont forget to remove bias
            for layer in reversed(self.layers):
                deltas = layer.computeDerivative(deltas, wgts)
                wgts = np.delete(layer.weights,0,0)

            # Update weights in the online learning fashion
            self._update_weights(self.learningRate)


    def classify(self, test_instance):
        # classified digit = maximum of the 10 output nodes
        self._feed_forward(test_instance)
        return (np.argmax(self.layers[-1].outp))
        

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
