#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2.0教程-保持和读取模型

# In[1]:


from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__


# 导入数据

# In[2]:


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# ## 1.定义一个模型

# In[5]:


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                 loss=keras.losses.sparse_categorical_crossentropy,
                 metrics=['accuracy'])
    return model
model = create_model()
model.summary()
    


# ## 2.checkpoint回调

# In[7]:


check_path = '106save/model.ckpt'
check_dir = os.path.dirname(check_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(check_path, 
                                                 save_weights_only=True, verbose=1)
model = create_model()
model.fit(train_images, train_labels, epochs=10,
         validation_data=(test_images, test_labels),
         callbacks=[cp_callback])


# In[9]:


get_ipython().system('ls {check_dir}')


# In[10]:


model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


# In[11]:


model.load_weights(check_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


# ## 3.设置checkpoint回调

# In[12]:


check_path = '106save02/cp-{epoch:04d}.ckpt'
check_dir = os.path.dirname(check_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True, 
                                                 verbose=1, period=5)  # 每5
model = create_model()
model.fit(train_images, train_labels, epochs=10,
         validation_data=(test_images, test_labels),
         callbacks=[cp_callback])


# In[14]:


get_ipython().system('ls {check_dir}')


# 载入最新版模型

# In[16]:


latest = tf.train.latest_checkpoint(check_dir)
print(latest)


# In[18]:


model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print('restored model accuracy: {:5.2f}%'.format(acc*100))


# ## 5.手动保持权重

# In[20]:


model.save_weights('106save03/manually_model.ckpt')
model = create_model()
model.load_weights('106save03/manually_model.ckpt')
loss, acc = model.evaluate(test_images, test_labels)
print('restored model accuracy: {:5.2f}%'.format(acc*100))


# ## 6.保持整个模型

# In[22]:


model = create_model()
model.fit(train_images, train_labels, epochs=10,
         validation_data=(test_images, test_labels),
         )
model.save('106save03.h5')


# In[23]:


new_model = keras.models.load_model('106save03.h5')
new_model.summary()


# In[24]:


loss, acc = model.evaluate(test_images, test_labels)
print('restored model accuracy: {:5.2f}%'.format(acc*100))


# ## 7.其他导出模型的方法

# In[26]:


import time
saved_model_path = "./saved_models/{}".format(int(time.time()))

tf.keras.experimental.export_saved_model(model, saved_model_path)
saved_model_path


# In[27]:


new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()


# In[28]:


# 该方法必须先运行compile函数
new_model.compile(optimizer=model.optimizer,  # keep the optimizer that was loaded
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[ ]:




