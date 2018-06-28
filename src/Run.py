#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    # ------------  NOTE  --------------
    # oneHot does not work: It makes not sense to have binary labeled data (targetDigit or notTargetDigit), 
    # but having a MLP with 10 output nodes and softmax function. It needs to be trained with digit labels, not binary ones!
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)
                                                    
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        hiddenLayerSizes=[65,30], # size of hidden layers, input and output layers sizes are constant
                                        learningRate=0.025, # learning rate
                                        epochs=50) # epochs
                                                              

    # Train the classifiers
    print("=========================")
    print("Training..")
    
    print("\nMLP has been training..")
    myMLPClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    mlpPred = myMLPClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    
    print("\nResult of the MLP:")
    #evaluator.printComparison(data.testSet, lrPred)    
    evaluator.printAccuracy(data.testSet, mlpPred)
    
    # Draw
    plot = PerformancePlot("MLP validation")
    plot.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)
    
    
if __name__ == '__main__':
    main()
