#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2.0教程-结构化数据分类
# 
# tensorflow2教程知乎专栏：https://zhuanlan.zhihu.com/c_1091021863043624960
# 
# 本教程展示了如何对结构化数据进行分类（例如CSV中的表格数据）。我们使用Keras定义模型，并将csv中各列的特征转化为训练的输入。 本教程包含一下功能代码：
# 
# - 使用Pandas加载CSV文件。
# - 构建一个输入的pipeline，使用tf.data批处理和打乱数据。
# - 从CSV中的列映射到用于训练模型的输入要素。
# - 使用Keras构建，训练和评估模型。

# In[2]:


from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
print(tf.__version__)


# ## 1.数据集
# 我们将使用克利夫兰诊所心脏病基金会提供的一个小数据集。 CSV中有几百行。 每行描述一个患者，每列描述一个属性。 我们将使用此信息来预测患者是否患有心脏病，该疾病在该数据集中是二元分类任务。
# 
# >Column| Description| Feature Type | Data Type
# >------------|--------------------|----------------------|-----------------
# >Age | Age in years | Numerical | integer
# >Sex | (1 = male; 0 = female) | Categorical | integer
# >CP | Chest pain type (0, 1, 2, 3, 4) | Categorical | integer
# >Trestbpd | Resting blood pressure (in mm Hg on admission to the hospital) | Numerical | integer
# >Chol | Serum cholestoral in mg/dl | Numerical | integer
# >FBS | (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) | Categorical | integer
# >RestECG | Resting electrocardiographic results (0, 1, 2) | Categorical | integer
# >Thalach | Maximum heart rate achieved | Numerical | integer
# >Exang | Exercise induced angina (1 = yes; 0 = no) | Categorical | integer
# >Oldpeak | ST depression induced by exercise relative to rest | Numerical | integer
# >Slope | The slope of the peak exercise ST segment | Numerical | float
# >CA | Number of major vessels (0-3) colored by flourosopy | Numerical | integer
# >Thal | 3 = normal; 6 = fixed defect; 7 = reversable defect | Categorical | string
# >Target | Diagnosis of heart disease (1 = true; 0 = false) | Classification | integer
# 

# ## 2.准备数据
# 使用pandas读取数据

# In[3]:


URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()


# 划分训练集验证集和测试集

# In[4]:


train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# 使用tf.data构造输入pipeline

# In[7]:


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)



# In[8]:


for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch )


# ## 3.tensorflow的feature column

# In[9]:


example_batch = next(iter(train_ds))[0]


# In[10]:


def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


# ### 数字列
# 特征列的输出成为模型的输入。 数字列是最简单的列类型。 它用于表示真正有价值的特征。 使用此列时，模型将从数据框中接收未更改的列值。

# In[11]:


age = feature_column.numeric_column("age")
demo(age)


# ### Bucketized列（桶列）
# 通常，您不希望将数字直接输入模型，而是根据数值范围将其值分成不同的类别。 考虑代表一个人年龄的原始数据。 我们可以使用bucketized列将年龄分成几个桶，而不是将年龄表示为数字列。 请注意，下面的one-hot描述了每行匹配的年龄范围。

# In[12]:


age_buckets = feature_column.bucketized_column(age, boundaries=[
    18, 25, 30, 35, 40, 50
])
demo(age_buckets)


# ### 类别列
# 在该数据集中，thal表示为字符串（例如“固定”，“正常”或“可逆”）。 我们无法直接将字符串提供给模型。 相反，我们必须首先将它们映射到数值。 类别列提供了一种将字符串表示为单热矢量的方法（就像上面用年龄段看到的那样）。 类别表可以使用categorical_column_with_vocabulary_list作为列表传递，或者使用categorical_column_with_vocabulary_file从文件加载。

# In[13]:


thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)


# ### 嵌入列
# 假设我们不是只有几个可能的字符串，而是每个类别有数千（或更多）值。 由于多种原因，随着类别数量的增加，使用单热编码训练神经网络变得不可行。 我们可以使用嵌入列来克服此限制。 嵌入列不是将数据表示为多维度的单热矢量，而是将数据表示为低维密集向量，其中每个单元格可以包含任意数字，而不仅仅是0或1.嵌入的大小是必须训练调整的参数。
# 
# 注：当分类列具有许多可能的值时，最好使用嵌入列。

# In[16]:


thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)


# ### 哈希特征列
# 表示具有大量值的分类列的另一种方法是使用categorical_column_with_hash_bucket。 此功能列计算输入的哈希值，然后选择一个hash_bucket_size存储桶来编码字符串。 使用此列时，您不需要提供词汇表，并且可以选择使hash_buckets的数量远远小于实际类别的数量以节省空间。
# 
# 注：该技术的一个重要缺点是可能存在冲突，其中不同的字符串被映射到同一个桶。

# In[17]:


thal_hashed = feature_column.categorical_column_with_hash_bucket('thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))


# ### 交叉功能列
# 将特征组合成单个特征（更好地称为特征交叉），使模型能够为每个特征组合学习单独的权重。 在这里，我们将创建一个与age和thal交叉的新功能。 请注意，crossed_column不会构建所有可能组合的完整表（可能非常大）。 相反，它由hashed_column支持，因此您可以选择表的大小。

# In[18]:


crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))


# ## 4.选择使用feature column

# In[19]:


feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)


# 构建特征层

# In[20]:


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# In[21]:


batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# ## 5.构建模型并训练

# In[22]:


model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds,epochs=5)


# 测试

# In[23]:


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)


# In[ ]:




