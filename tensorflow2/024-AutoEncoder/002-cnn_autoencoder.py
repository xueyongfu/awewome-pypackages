#!/usr/bin/env python
# coding: utf-8

# # Tensorflow2教程-卷积自编码器

# ![](https://camo.githubusercontent.com/c2b4e51b1ebacac0d5fae4796bff2572797cc385/687474703a2f2f6d692e656e672e63616d2e61632e756b2f70726f6a656374732f7365676e65742f696d616765732f7365676e65742e706e67)

# ## 1.导入数据

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
print(tf.__version__)


# In[2]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[4]:


x_train = tf.expand_dims(x_train.astype('float32'), -1) / 255.0
x_test = tf.expand_dims(x_test.astype('float32'),-1) / 255.0

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)


# ## 2.构造模型

# In[14]:



inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), name='inputs')
print(inputs.shape)
code = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
code = layers.MaxPool2D((2,2), padding='same')(code)
print(code.shape)
decoded = layers.Conv2D(16, (3,3), activation='relu', padding='same')(code)
decoded = layers.UpSampling2D((2,2))(decoded)
print(decoded.shape)
outputs = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(decoded)
print(outputs.shape)
auto_encoder = keras.Model(inputs, outputs)


# In[15]:


auto_encoder.compile(optimizer=keras.optimizers.Adam(),
                    loss=keras.losses.BinaryCrossentropy())
keras.utils.plot_model(auto_encoder, show_shapes=True)


# ## 3.训练与测试

# In[16]:


early_stop = keras.callbacks.EarlyStopping(patience=2, monitor='loss')
auto_encoder.fit(x_train,x_train, batch_size=64, epochs=1, validation_split=0.1,validation_freq=10,
                callbacks=[early_stop])


# In[19]:


import matplotlib.pyplot as plt
decoded = auto_encoder.predict(x_test)
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(tf.reshape(x_test[i+1],(28, 28)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(tf.reshape(decoded[i+1],(28, 28)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




