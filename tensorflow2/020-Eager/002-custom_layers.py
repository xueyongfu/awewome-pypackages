#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Tensorflow2.0教程-自定义层
# 
# tensorflow2.0建议使用tf.keras作为构建神经网络的高级API。 也就是说，大多数TensorFlow API都可用于eager执行模式。

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[2]:


get_ipython().system('pip install -q tensorflow==2.0.0-alpha0')
import tensorflow as tf
print(tf.__version__)


# ## 一、网络层layer的常见操作
# 通常机器学习模型可以表示为简单网络层的堆叠与组合，而tensorflow就提供了常见的网络层，为我们编写神经网络程序提供了便利。
# TensorFlow2推荐使用tf.keras来构建网络层，tf.keras来自原生keras，用其来构建网络具有更好的可读性和易用性。
# 
# 如，我们要构造一个简单的全连接网络，只需要指定网络的神经元个数

# In[ ]:


layer = tf.keras.layers.Dense(100)
# 也可以添加输入维度限制
layer = tf.keras.layers.Dense(100, input_shape=(None, 20))


# 可以在[文档](https://www.tensorflow.org/api_docs/python/tf/keras/layers)中查看预先存在的图层的完整列表。 它包括Dense，Conv2D，LSTM，BatchNormalization，Dropout等等。
# 
# 每个层都可以当作一个函数，然后以输入的数据作为函数的输入

# In[ ]:


layer(tf.ones([6, 6]))


# 同时我们也可以得到网络的变量、权重矩阵、偏置等 

# In[ ]:


print(layer.variables) # 包含了权重和偏置


# In[ ]:


print(layer.kernel, layer.bias)  # 也可以分别取出权重和偏置


# ## 二、实现自定义网络层
# 实现自己的层的最佳方法是扩展tf.keras.Layer类并实现：
# 
# - \__init__()函数，你可以在其中执行所有与输入无关的初始化
# 
# - build()函数，可以获得输入张量的形状，并可以进行其余的初始化
# 
# - call()函数，构建网络结构，进行前向传播
# 
# 实际上，你不必等到调用build()来创建网络结构，您也可以在\__init__()中创建它们。 但是，在build()中创建它们的优点是它可以根据图层将要操作的输入的形状启用后期的网络构建。 另一方面，在\__init__中创建变量意味着需要明确指定创建变量所需的形状。

# In[ ]:


class MyDense(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super(MyDense, self).__init__()
        self.n_outputs = n_outputs
    
    def build(self, input_shape):
        self.kernel = self.add_variable('kernel',
                                       shape=[int(input_shape[-1]),
                                             self.n_outputs])
    def call(self, input):
        return tf.matmul(input, self.kernel)
layer = MyDense(10)
print(layer(tf.ones([6, 5])))
print(layer.trainable_variables)


# ## 三、网络层组合
# 机器学习模型中有很多是通过叠加不同的结构层组合而成的，如resnet的每个残差块就是“卷积+批标准化+残差连接”的组合。
# 
# 在tensorflow2中要创建一个包含多个网络层的的结构，一般继承与tf.keras.Model类。
# 

# In[8]:


# 残差块
class ResnetBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetBlock, self).__init__(name='resnet_block')
        
        # 每个子层卷积核数
        filter1, filter2, filter3 = filters
        
        # 三个子层，每层1个卷积加一个批正则化
        # 第一个子层， 1*1的卷积
        self.conv1 = tf.keras.layers.Conv2D(filter1, (1,1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        # 第二个子层， 使用特点的kernel_size
        self.conv2 = tf.keras.layers.Conv2D(filter2, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        # 第三个子层，1*1卷积
        self.conv3 = tf.keras.layers.Conv2D(filter3, (1,1))
        self.bn3 = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs, training=False):
        
        # 堆叠每个子层
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        # 残差连接
        x += inputs
        outputs = tf.nn.relu(x)
        
        return outputs

resnetBlock = ResnetBlock(2, [6,4,9])
# 数据测试
print(resnetBlock(tf.ones([1,3,9,9])))
# 查看网络中的变量名
print([x.name for x in resnetBlock.trainable_variables])


# 如果模型是线性的，可以直接用tf.keras.Sequential来构造。

# In[9]:


seq_model = tf.keras.Sequential(
[
    tf.keras.layers.Conv2D(1, 1, input_shape=(None, None, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(2, 1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(3, 1),
    tf.keras.layers.BatchNormalization(),
    
])
seq_model(tf.ones([1,2,3,3]))


# In[ ]:




