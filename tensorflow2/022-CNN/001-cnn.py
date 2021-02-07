#!/usr/bin/env python
# coding: utf-8

# # tensorflow2-基础CNN网络
# ![](https://adeshpande3.github.io/assets/Cover.png)

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)


# ## 1.构造数据

# In[2]:



(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)


# In[4]:


import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()


# In[5]:


x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))


# ## 2.构造网络

# In[6]:


model = keras.Sequential()


# ### 卷积层
# ![](http://cs231n.github.io/assets/cnn/depthcol.jpeg)

# In[7]:


model.add(layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                        filters=32, kernel_size=(3,3), strides=(1,1), padding='valid',
                       activation='relu'))


# ### 池化层
# ![](http://cs231n.github.io/assets/cnn/maxpool.jpeg)

# In[8]:


model.add(layers.MaxPool2D(pool_size=(2,2)))


# ### 全连接层

# In[9]:


model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
# 分类层
model.add(layers.Dense(10, activation='softmax'))


# ## 3.模型配置

# In[10]:


model.compile(optimizer=keras.optimizers.Adam(),
             # loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
             loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()


# ## 4.模型训练

# In[11]:


history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)


# In[12]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()


# In[13]:


res = model.evaluate(x_test, y_test)


# In[ ]:




