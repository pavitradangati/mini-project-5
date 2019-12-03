import numpy as np

def multiclassLRPredict(model, x):
    # Simply predict the first class (Implement this)
    #ypred = model['classLabels'][0]*np.ones(numData)
    ypred = np.dot(model['w'], x)+model['b']
    yscores = ypred-np.max(ypred, axis=0)
    scores_exp = np.exp(yscores)
    probability = scores_exp/np.sum(scores_exp, axis=0)
    idxs = np.argmax(probability, axis=0)
    ypred = model['classLabels'][idxs]
    return ypred
