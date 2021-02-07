#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2教程-CNN-RNN结构用于图像处理

# ## 1.导入数据
# 使用cifar-10数据集
# ![](https://image.slidesharecdn.com/pycon2015-150913033231-lva1-app6892/95/pycon-2015-48-638.jpg?cb=1442115225)

# In[1]:


import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras


# In[2]:


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# In[3]:


y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


# In[4]:


print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)


# ## 2.简单的cnn-rnn结构

# In[5]:


from tensorflow.keras import layers
model = keras.Sequential()


# In[6]:


x_shape = x_train.shape
model.add(layers.Conv2D(input_shape=(x_shape[1], x_shape[2], x_shape[3]),
                       filters=32, kernel_size=(3,3), strides=(1,1), 
                       padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))


# In[7]:


print(model.output_shape)


# In[8]:


model.add(layers.Reshape(target_shape=(16*16, 32)))
model.add(layers.LSTM(50, return_sequences=False))

model.add(layers.Dense(10, activation='softmax'))


# In[9]:


model.compile(optimizer=keras.optimizers.Adam(),
             loss=keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])
model.summary()


# In[10]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(x_train, y_train, batch_size=32,epochs=5, validation_split=0.1)')


# In[20]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()


# In[12]:


res = model.evaluate(x_test,y_test)


# ## CNN和LSTM结果合并

# In[13]:


x_shape = x_train.shape
inn = layers.Input(shape=(x_shape[1], x_shape[2], x_shape[3]))
conv = layers.Conv2D(filters=32,kernel_size=(3,3), strides=(1,1),
                    padding='same', activation='relu')(inn)
pool = layers.MaxPool2D(pool_size=(2,2), padding='same')(conv)
flat = layers.Flatten()(pool)
dense1 = layers.Dense(64)(flat)


# In[14]:


reshape = layers.Reshape(target_shape=(x_shape[1]*x_shape[2], x_shape[3]))(inn)
lstm_layer = layers.LSTM(32, return_sequences=False)(reshape)
dense2 = layers.Dense(64)(lstm_layer)


# In[18]:


merged_layer = layers.concatenate([dense1, dense2])
outt = layers.Dense(10,activation='softmax')(merged_layer)
model = keras.Model(inputs=inn, outputs=outt)
model.compile(optimizer=keras.optimizers.Adam(),
             loss=keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])
model.summary()


# In[19]:


get_ipython().run_cell_magic('time', '', 'history2 = model.fit(x_train, y_train, batch_size=64,epochs=5, validation_split=0.1)')


# In[21]:


plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()


# In[ ]:




