#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2.0教程-张量极其操作

# ## 导入TensorFlow
# 运行tensorflow程序，需要导入tensorflow模块。
# 从TensorFlow 2.0开始，默认情况下会启用急切执行。 这为TensorFlow提供了一个更加互动的前端节。

# In[2]:


from __future__ import absolute_import, division, print_function
import tensorflow as tf


# ## 1 Tensors
# 张量是一个多维数组。 与NumPy ndarray对象类似，tf.Tensor对象具有数据类型和形状。 此外，tf.Tensors可以驻留在加速器内存中（如GPU）。 TensorFlow提供了丰富的操作库（tf.add，tf.matmul，tf.linalg.inv等），它们使用和生成tf.Tensors。 这些操作会自动转换原生Python类型，例如：

# In[4]:


print(tf.add(1,2))
print(tf.add([3,8], [2,5]))
print(tf.square(6))
print(tf.reduce_sum([7,8,9]))
print(tf.square(3)+tf.square(4))


# 每个Tensor都有形状和类型

# In[5]:


x = tf.matmul([[3], [6]], [[2]])
print(x)
print(x.shape)
print(x.dtype)


# NumPy数组和tf.Tensors之间最明显的区别是：
# 
# 张量可以由加速器内存（如GPU，TPU）支持。
# 张量是不可变的。

# NumPy兼容性
# 在TensorFlow tf.Tensors和NumPy ndarray之间转换很容易：
# 
# TensorFlow操作自动将NumPy ndarrays转换为Tensors。
# NumPy操作自动将Tensors转换为NumPy ndarrays。
# 使用.numpy（）方法将张量显式转换为NumPy ndarrays。 这些转换通常很容易的，因为如果可能，array和tf.Tensor共享底层内存表示。 但是，共享底层表示并不总是可行的，因为tf.Tensor可以托管在GPU内存中，而NumPy阵列总是由主机内存支持，并且转换涉及从GPU到主机内存的复制。

# In[7]:


import numpy as np
ndarray = np.ones([2,2])
tensor = tf.multiply(ndarray, 36)
print(tensor)
# 用np.add对tensorflow进行加运算
print(np.add(tensor, 1))
# 转换为numpy类型
print(tensor.numpy())


# ## 2 GPU加速
# 使用GPU进行计算可以加速许多TensorFlow操作。 如果没有任何注释，TensorFlow会自动决定是使用GPU还是CPU进行操作 - 如有必要，可以复制CPU和GPU内存之间的张量。 由操作产生的张量通常由执行操作的设备的存储器支持，例如：

# In[8]:


x = tf.random.uniform([3, 3])
print('Is GPU availabel:')
print(tf.test.is_gpu_available())
print('Is the Tensor on gpu #0:')
print(x.device.endswith('GPU:0'))


# **设备名称**
# 
# Tensor.device属性提供托管张量内容的设备的完全限定字符串名称。 此名称编码许多详细信息，例如正在执行此程序的主机的网络地址的标识符以及该主机中的设备。 这是分布式执行TensorFlow程序所必需的。 如果张量位于主机上的第N个GPU上，则字符串以GPU结尾：<N>。

# **显式设备放置(Placement)**
# 
# 在TensorFlow中，放置指的是如何分配（放置）设备以执行各个操作。 如上所述，如果没有提供明确的指导，TensorFlow会自动决定执行操作的设备，并在需要时将张量复制到该设备。 但是，可以使用tf.device上下文管理器将TensorFlow操作显式放置在特定设备上，例如：

# In[9]:


import time
def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    result = time.time() - start
    print('10 loops: {:0.2}ms'.format(1000*result))
    
# 强制使用CPU
print('On CPU:')
with tf.device('CPU:0'):
    x = tf.random.uniform([1000, 1000])
    # 使用断言验证当前是否为CPU0
    assert x.device.endswith('CPU:0')
    time_matmul(x)    

# 如果存在GPU,强制使用GPU
if tf.test.is_gpu_available():
    print('On GPU:')
    with tf.device.endswith('GPU:0'):
        x = tf.random.uniform([1000, 1000])
    # 使用断言验证当前是否为GPU0
    assert x.device.endswith('GPU:0')
    time_matmul(x)  


# ## 3 数据集
# 本节使用tf.data.Dataset API构建管道，以便为模型提供数据。 tf.data.Dataset API用于从简单，可重复使用的部分构建高性能，复杂的输入管道，这些部分将为模型的培训或评估循环提供支持。

# **创建源数据集**
# 使用其中一个工厂函数（如Dataset.from_tensors，Dataset.from_tensor_slices）或使用从TextLineDataset或TFRecordDataset等文件读取的对象创建源数据集。 有关详细信息，请参阅TensorFlow数据集指南。

# In[13]:


# 从列表中获取tensor
ds_tensors = tf.data.Dataset.from_tensor_slices([6,5,4,3,2,1])
# 创建csv文件
import tempfile
_, filename = tempfile.mkstemp()
print(filename)

with open(filename, 'w') as f:
    f.write("""Line 1
Line 2
Line 3""")
# 获取TextLineDataset数据集实例
ds_file = tf.data.TextLineDataset(filename)


# **应用转换**
# 
# 使用map，batch和shuffle等转换函数将转换应用于数据集记录。
# 

# In[14]:


ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)


# **迭代**
# 
# tf.data.Dataset对象支持迭代循环记录：

# In[15]:


print('ds_tensors中的元素：')
for x in ds_tensors:
    print(x)
# 从文件中读取的对象创建的数据源
print('\nds_file中的元素：')
for x in ds_file:
    print(x)


# In[ ]:




