#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2.0教程-回归

# 
# tensorflow2教程知乎专栏：https://zhuanlan.zhihu.com/c_1091021863043624960
# 
# 在回归问题中，我们的目标是预测连续值的输出，如价格或概率。 
# 我们采用了经典的Auto MPG数据集，并建立了一个模型来预测20世纪70年代末和80年代初汽车的燃油效率。 为此，我们将为该模型提供该时段内许多汽车的描述。 此描述包括以下属性：气缸，排量，马力和重量。

# In[2]:


from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# ## 1.Auto MPG数据集
# 

# 获取数据

# In[4]:


dataset_path = keras.utils.get_file('auto-mpg.data',
                                   'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')

print(dataset_path)


# 使用pandas读取数据

# In[5]:


column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                         na_values='?', comment='\t',
                         sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
dataset.tail()


# ## 2.数据预处理
# ### 清洗数据

# In[6]:


print(dataset.isna().sum())
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()


# ### 划分训练集和测试集

# In[8]:


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# ### 检测数据
# 
# 观察训练集中几对列的联合分布。

# In[10]:


sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")


# 整体统计数据：

# In[11]:


train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats


# ### 取出标签

# In[12]:


train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# ### 标准化数据
# 最好使用不同比例和范围的特征进行标准化。 虽然模型可能在没有特征归一化的情况下收敛，但它使训练更加困难，并且它使得结果模型依赖于输入中使用的单位的选择。

# In[13]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# ## 3.构建模型

# In[14]:


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['mae', 'mse'])
    return model


# In[15]:


model = build_model()
model.summary()


# In[16]:


example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# ## 4.训练模型

# In[17]:


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


# 查看训练记录

# In[18]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[19]:


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()


plot_history(history)


# 使用early stop

# In[20]:


model = build_model()


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# 测试

# In[21]:


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# # 5.预测

# In[22]:


test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


# In[23]:


error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")


# In[ ]:




