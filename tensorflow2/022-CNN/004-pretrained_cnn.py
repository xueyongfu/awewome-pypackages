#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2教程-使用预训练模型

# In[1]:


#! pip install Pillow
get_ipython().system(" curl 'https://gfp-2a3tnpzj.stackpathdns.com/wp-content/uploads/2016/07/Dachshund-600x600.jpg' --output dog.jpg")
get_ipython().system(' ls')


# In[2]:


import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50


# In[3]:


img = image.load_img('dog.jpg')
print(image.img_to_array(img).shape)
img


# ## 1.导入模型
# 目前看使用模型：
# ### Import model
# - Currently, seven models are supported
#     - Xception
#     - VGG16
#     - VGG19
#     - ResNet50
#     - InceptionV3
#     - InceptionResNetV2
#     - MobileNet
#     - MobileNetV2
#     - DenseNet
#     - nasnet
# 

# In[4]:


model = resnet50.ResNet50(weights='imagenet')


# In[5]:


img = image.load_img('dog.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
print(img.shape)


# ## 2.模型预测

# In[6]:


pred_class = model.predict(img)


# In[7]:


n = 10
top_n = resnet50.decode_predictions(pred_class, top=n)
for c in top_n[0]:
    print(c)


# In[8]:


# img = image.load_img('dog.jpg')
# img = image.img_to_array(img)
# print(img.shape)
img = resnet50.preprocess_input(img)
print(img.shape)


# In[9]:


pred_class = model.predict(img)
n = 10
top_n = resnet50.decode_predictions(pred_class, top=n)
for c in top_n[0]:
    print(c)

