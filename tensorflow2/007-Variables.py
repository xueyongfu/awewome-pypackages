#!/usr/bin/env python
# coding: utf-8

# # TensorFlow教程-Variables

# 创建一个变量

# In[7]:


import tensorflow as tf
my_var = tf.Variable(tf.ones([2,3]))
print(my_var)
try:
    with tf.device("/device:GPU:0"):
        v = tf.Variable(tf.zeros([10, 10]))
        print(v)
except:
    print('no gpu')


# 使用变量
# 

# In[8]:


a = tf.Variable(1.0)
b = (a+2) *3
print(b)


# In[9]:


a = tf.Variable(1.0)
b = (a.assign_add(2)) *3
print(b)


# 变量跟踪
# 

# In[11]:


class MyModuleOne(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(x) for x in range(10)]
    
class MyOtherModule(tf.Module):
    def __init__(self):
        self.m = MyModuleOne()
        self.v = tf.Variable(10.0)
    
m = MyOtherModule()
print(m.variables)
len(m.variables) 


# In[ ]:




