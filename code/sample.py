import numpy as np 
from collections import Counter
"""
print(np.arctan2(1,0)*180/np.pi)
arr = [[[1,2],[1,2]],[[1,2],[1,2]]]
arr1 = [[[1,2],[1,2],[1,2]]]
print(arr+arr1)

a = np.array([1,2])
b = np.ones((3,4))
c = a[:, np.newaxis, np.newaxis]-b
print(a.shape)
print(b.shape)
print(c.shape)
print(c)


arr = np.array([[[1,2,12],[3,4,13]],[[5,6,14],[7,8,15]]])
d = arr.reshape((4,-1))
print(arr.shape)
print(d)
print(d.reshape(2,2,-1))
print(arr)

"""

arr = np.array([[[1,2], [3,4], [5,6]], [[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]]])
print(arr.shape)
print(arr)
newArr = np.transpose(arr)
print(newArr.shape)
print(newArr)
#print(np.transpose(arr))


a = np.array([[1,2,3],[3,4,5]])
b = np.array([[1,2,3],[3,4,5],[6,7,8],[9,10,11]])
print(c)
print(a.shape, b.shape, c.shape)
