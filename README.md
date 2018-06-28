# NNPraktikum
Built on top of the coding framework for KIT Neural Network Praktikum
See https://github.com/thanhleha-kit/NNPraktikum

NOTE: oneHot implemented by the provided framework is not supported because it does not make really sense in the context of this MLP. Since the given output layer has size 10 and we should use the softmax function, it does not make sense to train the MLP with binary target values (target digit or not), instead we train it with the labeled digit itself.
When running the network, the first 30 target labels and network outputs are printed for visual comparision (verbose).

## Python version
Built on python 2.7

