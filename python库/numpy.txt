#返回数组中元素出现的次数

import numpy as np 
lable = [1,2,3,3,2,1,6]
np.unique(lable, return_counts = True)

# np.eye()产生对角矩阵

print(np.eye(6))

# stack()
import numpy as np

a=[[1,2,3],
   [4,5,6]]

print(np.stack(a,axis=0))
print(np.stack(a,axis=1))

# vstack()水平stack, 直接水平放置
# hstack()垂直stack, 直接垂直放置

a=[1,2,3]
b=[4,5,6]

print(np.hstack((a,b)))
print(np.vstack((a,b)))

# np.concatenate

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print(np.concatenate((a,b),axis=0))   #相当于np.vstack((a,b))
print(np.concatenate((a,b),axis=1))    #相当于np.hstack((a,b))
print(np.concatenate((a,b),axis=-1))   #axis=-1表示最后一个维度

import operator
import functools

a = [[1,2,3], [4,6], [7,8,9,8]]
functools.reduce(operator.add, a)

from itertools import chain

b=[[1,2,3], [5,8], [7,8,9]]

list(chain(*b))

a=[[1,2,3], [5,8], [7,8,9]]

a= eval('['+str(a).replace(' ','').replace('[','').replace(']','')+']')
print(a)


























