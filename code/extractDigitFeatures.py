import numpy as np
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
import math
from collections import Counter
# EXTRACTDIGITFEATURES extracts features from digit images
#   features = extractDigitFeatures(x, featureType) extracts FEATURES from images
#   images X of the provided FEATURETYPE. The images are assumed to the of
#   size [W H 1 N] where the first two dimensions are the width and height.
#   The output is of size [D N] where D is the size of each feature and N
#   is the number of images. 
def extractDigitFeatures(x, featureType):
    
    if featureType == 'pixel':
        features = pixelFeatures(x)  # implement this
    elif featureType == 'hog':
        features = hogFeatures(x)  # implement this
    elif featureType == 'lbp':
        features = lbp(x)  # implement this
    """
    #feature normalization
    features = np.sqrt(features)
    #L2-normalization
    features = features/np.sqrt(np.sum(features**2, axis=0))
    print('features shape', features.shape)
    """
    return features

def getPatch(img1, ws):
    padding = int(math.floor(ws/2))
    patch1 = np.zeros((img1.shape[0], img1.shape[1], img1.shape[2], ws*ws))
    newImg1 = np.zeros((img1.shape[0]+2*padding, img1.shape[1]+2*padding))
    newImg1[padding:padding+img1.shape[0], padding:padding+img1.shape[1]] = img1[:,:]
    for i in range(padding, img1.shape[0]+padding):
        for j in range(padding, img1.shape[1]+padding):
            patch1[i-padding][j-padding] = (newImg1[i-padding:i+padding+1, j-padding:j+padding+1]).flatten()
    return patch1

def lbp(x):
    padding = 1
    w,h,im = x.shape
    paddedImage = np.zeros((w+2*padding, h+2*padding, im))
    paddedImage[padding:padding+w, padding:padding+h, :] = x[:,:,:]
    binaryBit = np.array([1,2,4,8,0,16,32,64,128])
    featureImage = np.zeros((w,h,im))
    for i in range(padding, w+padding):
        for j in range(padding, h+padding):
            patch = (paddedImage[i-padding:i+padding+1, j-padding:j+padding+1, :]).reshape(9,-1)
            patchDiff = patch - patch[4, :]
            patchDiff[patchDiff>=0] = 1
            patchDiff[patchDiff<0] = 0
            patchDiff[4,:]=0
            binRep = binaryBit[:, np.newaxis]* patchDiff 
            featureImage[i-padding, j-padding, :] = np.sum(binRep, axis=0) 
    featureImage = featureImage.reshape((w*h, -1))
    featureLBP = np.zeros((256, im))
    for i in range(im):
        c = Counter(featureImage[:,i])
        c = list(Counter(featureImage[:,i]).items())
        for k in c:
            featureLBP[int(k[0])][i] = k[1]
    return featureLBP

def getHistograms(x, magnitude, angles, numOri, binSize):
    xflatten = angles.reshape((x.shape[0]*x.shape[1], -1))
    magFlatten = magnitude.reshape((x.shape[0]*x.shape[1], -1))
    oriRange = np.linspace(-180,180, numOri+1)
    oriSize = np.abs(oriRange[0]-oriRange[1])
    dist = np.abs(oriRange[:, np.newaxis, np.newaxis]-xflatten)
    mask = np.copy(dist)
    mask[mask<45]=1
    mask[mask>=45]=0
    temp = np.abs(oriSize-dist)*mask
    for i in range(binSize*binSize):
        dist[:,i,:] = temp[:,i,:]*magFlatten[i,:]
    HoG = np.sum(dist, axis=1)
    return HoG

def hogFeatures(x):
    #Mapping pixel intensities by applying a non linearity
    #x = np.log(x)
    #x = x**0.8
    #plt.imshow(x[:,:,0])
    #plt.show()
    binSize = 4
    numOri = 8
    gx = sobel(x,axis=0,mode='constant')
    gy = sobel(x,axis=1, mode='constant')
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy,gx) *180/np.pi
    hog = None
    for i in range(int(x.shape[0]/binSize)):
        for j in range(int(x.shape[1]/binSize)):
            startx, endx = i*binSize, (i+1) * binSize
            starty, endy = j*binSize, (j+1) * binSize
            newHog = getHistograms(x[startx:endx, starty:endy, :], magnitude[startx:endx, starty:endy, :], angle[startx:endx, starty:endy,:], numOri, binSize)
            if hog is None:
                hog = newHog
            else:
                hog = np.concatenate((hog, newHog), axis=0)
    return hog

def pixelFeatures(x):
    return x.reshape((x.shape[0]*x.shape[1], -1))

def zeroFeatures(x):
    return np.zeros((10, x.shape[2]))

