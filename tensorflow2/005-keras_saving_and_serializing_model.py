#!/usr/bin/env python
# coding: utf-8

# # tensorflow2教程-keras模型保持和序列化
# 
# ## 1.保持序列模型和函数模型

# In[1]:


# 构建一个简单的模型并训练
from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)

predictions = model.predict(x_test)


# ### 1.1保持全模型
# 可以对整个模型进行保存，其保持的内容包括：
# - 该模型的架构
# - 模型的权重（在训练期间学到的）
# - 模型的训练配置（你传递给编译的），如果有的话
# - 优化器及其状态（如果有的话）（这使您可以从中断的地方重新启动训练）
# 

# In[2]:


import numpy as np
model.save('the_save_model.h5')
new_model = keras.models.load_model('the_save_model.h5')
new_prediction = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_prediction, atol=1e-6) # 预测结果一样


# ### 1.2 保持为SavedModel文件

# In[4]:


keras.experimental.export_saved_model(model, 'saved_model')
new_model = keras.experimental.load_from_saved_model('saved_model')
new_prediction = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_prediction, atol=1e-6) # 预测结果一样


# ### 1.3仅保存网络结构
# 仅保持网络结构，这样导出的模型并未包含训练好的参数

# In[6]:


config = model.get_config()
reinitialized_model = keras.Model.from_config(config)
new_prediction = reinitialized_model.predict(x_test)
assert abs(np.sum(predictions-new_prediction)) >0


# 也可以使用json保存网络结构

# In[7]:


json_config = model.to_json()
reinitialized_model = keras.models.model_from_json(json_config)
new_prediction = reinitialized_model.predict(x_test)
assert abs(np.sum(predictions-new_prediction)) >0


# ### 1.4仅保存网络参数

# In[8]:


weights = model.get_weights()
model.set_weights(weights)


# In[12]:


# 可以把结构和参数保存结合起来
config = model.get_config()
weights = model.get_weights()
new_model = keras.Model.from_config(config) # config只能用keras.Model的这个api
new_model.set_weights(weights)
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)


# ### 1.5完整的模型保持方法

# In[15]:


json_config = model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(json_config)

model.save_weights('path_to_my_weights.h5')

with open('model_config.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('path_to_my_weights.h5')

new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)


# In[16]:


# 当然也可以一步到位
model.save('path_to_my_model.h5')
del model
model = keras.models.load_model('path_to_my_model.h5')


# ### 1.6保存网络权重为SavedModel格式

# In[18]:


model.save_weights('weight_tf_savedmodel')
model.save_weights('weight_tf_savedmodel_h5', save_format='h5')


# ### 1.7子类模型参数保存
# 子类模型的结构无法保存和序列化，只能保持参数

# In[19]:


# 构建模型
class ThreeLayerMLP(keras.Model):
  
    def __init__(self, name=None):
        super(ThreeLayerMLP, self).__init__(name=name)
        self.dense_1 = layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(64, activation='relu', name='dense_2')
        self.pred_layer = layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.pred_layer(x)

def get_model():
    return ThreeLayerMLP(name='3_layer_mlp')

model = get_model()


# In[20]:


# 训练模型
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)


# In[21]:


# 保持权重参数
model.save_weights('my_model_weights', save_format='tf')

# 输出结果，供后面对比

predictions = model.predict(x_test)
first_batch_loss = model.train_on_batch(x_train[:64], y_train[:64])


# In[24]:


# 读取保存的模型参数
new_model = get_model()
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop())

#new_model.train_on_batch(x_train[:1], y_train[:1])

new_model.load_weights('my_model_weights')

new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)


new_first_batch_loss = new_model.train_on_batch(x_train[:64], y_train[:64])
assert first_batch_loss == new_first_batch_loss


# In[ ]:




