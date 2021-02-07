#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2.0教程-自动求导
# 
# 这节我们会介绍使用tensorflow2自动求导的方法。

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
# !pip uninstall tensorflow
#!pip install tensorflow==2.0.0-alpha
import tensorflow as tf
print(tf.__version__)


# In[ ]:





# ## 一、Gradient tapes
# tensorflow 提供tf.GradientTape api来实现自动求导功能。只要在tf.GradientTape()上下文中执行的操作，都会被记录与“tape”中，然后tensorflow使用反向自动微分来计算相关操作的梯度。
# 
# 

# In[2]:


x = tf.ones((2,2))

# 需要计算梯度的操作
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y,y)
# 计算z关于x的梯度
dz_dx = t.gradient(z, x)
print(dz_dx)


# 也可以输出对中间变量的导数

# In[3]:


# 梯度求导只能每个tape一次
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y,y)
    
dz_dy = t.gradient(z, y)
print(dz_dy)


# 默认情况下GradientTape的资源会在执行tf.GradientTape()后被释放。如果想多次计算梯度，需要创建一个持久的GradientTape。

# In[4]:


with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
    
dz_dx = t.gradient(z,x)
print(dz_dx)
dz_dy = t.gradient(z, y)
print(dz_dy)


# ## 二、记录控制流
# 因为tapes记录了整个操作，所以即使过程中存在python控制流（如if， while），梯度求导也能正常处理。

# In[6]:


def f(x, y):
    output = 1.0
    # 根据y的循环
    for i in range(y):
        # 根据每一项进行判断
        if i> 1 and i<5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
        # 返回梯度
        return t.gradient(out, x)
# x为固定值
x = tf.convert_to_tensor(2.0)

print(grad(x, 6))
print(grad(x, 5))
print(grad(x, 4))


# ## 三、高阶梯度
# GradientTape上下文管理器在计算梯度的同时也会保持梯度，所以GradientTape也可以实现高阶梯度计算，

# In[9]:


x = tf.Variable(1.0)

with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * x * x
    dy_dx = t2.gradient(y, x)
    print(dy_dx)
d2y_d2x = t1.gradient(dy_dx, x)
print(d2y_d2x)


# In[ ]:




