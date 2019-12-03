import numpy as np

from multiclassLRTrain import multiclassLRTrain

def trainModel(x, y):
    param = {}
    param['lambda'] = 0.008 # Regularization term
    param['maxiter'] = 1000     # Number of iterations
    param['eta'] = 0.01 # Learning rate

    return multiclassLRTrain(x, y, param)


""" Testset=2
Digits-normal.mat: LBP Features sqrt
lambda=0.006, eta=0.15 Accuracy=50.40

HOG Features: sqrt
lambda=0.0075, eta=0.18 accuracy=97.60

Pixel Features: Square root normalisation
lambda=0.0035, eta=0.25, accuracy=84.80
Mean normalisation:
l=0.0030, eta=0.275, acc=86
"""