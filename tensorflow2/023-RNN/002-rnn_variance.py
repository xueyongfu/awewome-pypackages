#!/usr/bin/env python
# coding: utf-8

# # Tensorflow2教程-rnn变体

# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers


# ## 1.导入数据

# In[ ]:


num_words = 30000
maxlen = 200


# In[ ]:


(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)


# In[ ]:


x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding='post')


# In[ ]:



print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)


# ## 1.LSTM

# In[9]:


def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model
model = lstm_model()
model.summary()


# In[11]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)')


# In[14]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()


# ## 3.GRU

# In[15]:


def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        layers.GRU(32, return_sequences=True),
        layers.GRU(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model
model = lstm_model()
model.summary()


# In[16]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)')


# In[17]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()


# ## 4.CuDNN LSTM

# In[20]:


def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        layers.CuDNNLSTM(32, return_sequences=True),
        layers.CuDNNLSTM(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model
model = lstm_model()
model.summary()
# tf2版没有


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)')


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()


# ## 5.CuDNN GRU

# In[ ]:


def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        layers.CuDNNGRU(32, return_sequences=True),
        layers.CuDNNGRU(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model
model = lstm_model()
model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)')


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()


# In[ ]:


layers.


# In[ ]:




