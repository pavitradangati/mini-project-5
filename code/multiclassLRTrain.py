import numpy as np

def multiclassLRTrain(x, y, param):

    classLabels = np.unique(y)
    numClass = classLabels.shape[0] #10
    numFeats = x.shape[0] #F
    numData = x.shape[1] #N
    print(classLabels, numClass, numFeats, numData)
    model = {}
    model['w'] = np.random.randn(numClass, numFeats)*0.01 #10xF
    #model['b'] = np.random.rand(numClass,1) #10
    model['b'] = np.ones((numClass,1))
    model['classLabels'] = classLabels
    #print('w shape', model['w'].shape)
    #print('b shape', model['b'].shape)
    #print('c shape', model['classLabels'])
    for i in range(param['maxiter']):
        yscores = np.dot(model['w'], x)+model['b']
        yscores = yscores-np.max(yscores, axis=0)
        scores_exp = np.exp(yscores)
        probability = scores_exp/np.sum(scores_exp, axis=0)
        logLikelihood = np.log(np.sum(scores_exp, axis=0))-yscores[y, range(numData)]
        loss = (np.sum(logLikelihood)/numData) + param['lambda']*np.sum(model['w']*model['w'])
        probability[y, range(numData)] = probability[y,range(numData)]-1
        probability = probability/numData
        dLdW = np.dot(probability, x.T) + (2*param['lambda']*model['w'])
        dLdB = np.sum(probability, axis=1).reshape(-1,1)
        model['w']-=param['eta']*dLdW
        model['b']-=param['eta']*dLdB
        #print(i, loss)
    return model

