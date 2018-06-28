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
    # oneHot = False, as the framwork provided implements binary one-hot encoding (is it a 7 = True/False)  
    # Our targets are one-of-k encoded (e.g. 1= (0,1,0,0,0,0,0,0,0)). Network predicts the exact number on the picture not just 7 = True/False
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)
                                                    
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        hiddenLayerSizes=[65,30], # size of hidden layers, input and output layers sizes are constant
                                        learningRate=0.028, # learning rate
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
