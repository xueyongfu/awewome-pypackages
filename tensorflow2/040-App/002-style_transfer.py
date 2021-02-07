#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2教程-神经风格转移

# 本教程使用深度学习以另一个图像的风格组成一个图像（。这被称为神经风格转移，并且该技术在艺术风格的神经算法（Gatys等人）中概述。
# 
# 神经风格转移是一种优化技术，用于拍摄两个图像 - 内容图像和风格参考图像（如着名画家的作品） - 并将它们混合在一起，使输出图像看起来像内容图像，但“画”在风格参考图像的风格。
# 
# 这是通过优化输出图像以匹配内容图像的内容统计和样式参考图像的样式统计来实现的。使用卷积网络从图像中提取这些统计数据。
# 
# 例如，让我们拍摄一只这只乌龟的照片和Wassily Kandinsky的作品7：
# 
# ![](https://tensorflow.org/beta/tutorials/generative/images/Green_Sea_Turtle_grazing_seagrass.jpg)
# 
# 绿海龟的形象-By P.Lindgren [CC BY-SA 3.0]（（https://creativecommons.org/licenses/by-sa/3.0），来自维基共享资源
# 
# ![](https://tensorflow.org/beta/tutorials/generative/images/kadinsky.jpg)
# 
# 如果康定斯基决定用这种风格专门描绘这只海龟的照片，那会是什么样子？像这样的东西？
# ![](https://tensorflow.org/beta/tutorials/generative/images/kadinsky-turtle.png)
# 

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


# In[2]:


import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import time
import functools


# 下载图像并选择样式图像和内容图像：

# In[3]:


content_path = tf.keras.utils.get_file('turtle.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


# ### 可视化输入
# 定义一个加载图像的函数，并将其最大尺寸限制为512像素。

# In[4]:


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# 创建一个简单的函数来显示图像：

# In[5]:


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


# In[6]:


content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')


# ### 定义内容和样式表示
# 使用模型的中间层来获取图像的内容和样式表示。从网络的输入层开始，前几个图层激活表示边缘和纹理等低级特征。当逐步浏览网络时，最后几层代表更高级别的特征 - 对象部分，如轮子或眼睛。在这种情况下，使用的是VGG19网络架构，这是一种预训练的图像分类网络。这些中间层是从图像中定义内容和样式的表示所必需的。对于输入图像，尝试匹配这些中间层的相应样式和内容目标表示。
# 
# 加载VGG19并在我们的图像上测试运行它以确保正确使用它：
# 
# 

# In[7]:


x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape


# In[8]:


predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]


# 现在加载VGG19没有分类层的网络结构，并列出图层名称

# In[9]:


vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

print()
for layer in vgg.layers:
    print(layer.name)


# 从网络中选择中间层以表示图像的样式和内容：

# In[10]:


# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# 风格和内容的中间层
# 那么为什么我们预训练的图像分类网络中的这些中间输出允许我们定义样式和内容表示？
# 
# 在高层次上，为了使网络执行图像分类（该网络已被训练过），它必须理解图像。这需要将原始图像作为输入像素并构建内部表示，该原始图像将原始图像像素转换为对图像中存在的特征的复杂理解。
# 
# 这也是卷积神经网络能够很好地概括的原因：它们能够捕获不变性并定义类别（例如猫与狗）中的特征，这些特征与背景噪声和其他麻烦无关。因此，在将原始图像馈送到模型和输出分类标签之间的某处，该模型用作复杂的特征提取器。通过访问模型的中间层，可以描述输入图像的内容和样式。
# 
# 

# ### 建立模型
# tf.keras.applications设计中的网络使您可以使用Keras功能API轻松提取中间层值。
# 
# 要使用功能API定义模型，请指定输入和输出：
# 
# model = Model(inputs, outputs)
# 
# 以下函数构建一个VGG19模型，该模型返回中间层输出列表：

# In[11]:


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


# 并创建模型：

# In[12]:


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()


# ### 计算风格
# 图像的内容由中间特征图的值表示。
# 
# 事实证明，图像的风格可以通过不同特征图上的平均值和相关性来描述。计算包含此信息的Gram矩阵，方法是在每个位置使用特征向量的外积，并在所有位置对该外积进行平均。可以针对特定图层计算此Gram矩阵，如下所示：
# ![Screenshot%20from%202019-07-28%2021-11-00.png](attachment:Screenshot%20from%202019-07-28%2021-11-00.png)
# 这可以使用以下tf.linalg.einsum函数简洁地实现

# In[13]:


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


# ### 提取样式和内容
# 构建一个返回样式和内容张量的模型。
# 

# In[14]:


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}


# 在图像上调用时，此模型返回以下内容的克数矩阵（样式）style_layers和内容content_layers：

# In[15]:


extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']

print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())


# ### 运行梯度下降
# 使用此样式和内容提取器，您现在可以实现样式传输算法。通过计算图像输出相对于每个目标的均方误差来做到这一点，然后取这些损失的加权和。
# 
# 设置样式和内容目标值：

# In[16]:


style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']


# 定义一个 tf.Variable以包含要优化的图像。要快速完成此操作，请使用内容图像（tf.Variable必须与内容图像的形状相同）对其进行初始化：

# In[17]:


image = tf.Variable(content_image)


# In[18]:


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# 创建一个优化器。本文推荐LBFGS，但也Adam可以正常工作：

# In[19]:


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


# 为了优化这一点，使用两个损失的加权组合来获得总损失：

# In[20]:


style_weight=1e-2
content_weight=1e4


# In[21]:


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


# 使用tf.GradientTape更新的图像。

# In[22]:


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# 测试

# In[23]:


train_step(image)
train_step(image)
train_step(image)
plt.imshow(image.read_value()[0])


# 花更多的时间进行优化，就会获得更好的结果

# In[24]:


import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
    display.clear_output(wait=True)
    imshow(image.read_value())
    plt.title("Train step: {}".format(step))
    plt.show()

end = time.time()
print("Total time: {:.1f}".format(end-start))


# ### 总变异损失
# 这个基本实现的一个缺点是它会产生大量的高频伪像。 使用图像的高频分量上的显式正则化项来减少这些。 在样式转移中，这通常称为总变异损失：

# In[26]:


def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var


# In[27]:


x_deltas, y_deltas = high_pass_x_y(content_image)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

plt.subplot(2,2,2)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2,2,3)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

plt.subplot(2,2,4)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")


# 这显示了高频分量如何增加。
# 
# 而且，该高频分量基本上是边缘检测器。您可以从Sobel边缘检测器获得类似的输出，例如：

# In[28]:


plt.figure(figsize=(14,10))

sobel = tf.image.sobel_edges(content_image)
plt.subplot(1,2,1)
imshow(clip_0_1(sobel[...,0]/4+0.5), "Horizontal Sobel-edges")
plt.subplot(1,2,2)
imshow(clip_0_1(sobel[...,1]/4+0.5), "Vertical Sobel-edges")


# 与此相关的正则化损失是值的平方和：

# In[29]:


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)


# ### 重新运行优化
# 设定total_variation_loss的权重：
# 
# 

# In[30]:


total_variation_weight=1e8


# 现在将它包含在train_step函数中：

# In[31]:


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight*total_variation_loss(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# 重新初始化变量

# In[32]:


image = tf.Variable(content_image)


# 并运行优化

# In[ ]:


import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
    display.clear_output(wait=True)
    imshow(image.read_value())
    plt.title("Train step: {}".format(step))
    plt.show()

end = time.time()
print("Total time: {:.1f}".format(end-start))


# In[ ]:




