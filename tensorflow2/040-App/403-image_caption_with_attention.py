#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2教程-基于注意力的图片描述生成

# 图片生成描述，就是给定一张图片，生成相应的文字描述。如下图的生成描述为“冲浪者在波浪上滑行”。
# ![](https://tensorflow.org/images/surf.jpg)
# 下面，我们将基于注意力机制来构建描述生成模型。
# ![](https://tensorflow.org/images/imcap_prediction.png)

# 该模型来自论文：The model architecture is similar to Show, Attend and Tell: Neural Image Caption Generation with Visual Attention。
# 下面，我们将会使用MS-COCO数据，预处理和缓存图像子集，训练编码器-解码器模型，并使用训练模型生成新的图像描述。
# 在教程中我们将使用大约20000个图像的30000个字幕

# In[1]:


import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle


# ## 下载并准备MS-COCO数据集
# MS-COCO数据集包含82000个图像，每个图像至少有5个不同的标题注释。（ps，文件比较大有13G最好提前下载）

# In[6]:


annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

name_of_zip = 'train2014.zip'
if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
  image_zip = tf.keras.utils.get_file(name_of_zip,
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip)+'/train2014/'
else:
  PATH = os.path.abspath('.')+'/train2014/'


# # 选择训练集
# 为了加快训练，我们这边选择30000个字幕及其对应的图片作为训练集。如果想提高模型能力可以使用更多的数据。
# 
# 

# In[7]:


# 读取字幕文件
with open(annotation_file, 'r') as f:
    annotations = json.load(f)
    
# 读字幕及对应图片
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] +' <end>'
    image_id = annot['image_id']
    
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
    
    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)
# 打乱数据
train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector,
                                        random_state=1)
# 选取30000个字幕
num_examples = 30000
train_captions = train_captions[:30000]
img_name_vector = img_name_vector[:30000]


# In[8]:


len(train_captions), len(all_captions)


# # 使用InceptionV3预处理图像
# 接下来，您将使用InceptionV3（在Imagenet上预先训练）来对每个图像进行分类。
# 
# 首先，通过以下方式将图像转换为InceptionV3的预期格式：*将图像大小调整为299px×299px * 使用preprocess_input方法对图像进行预处理以对图像进行标准化，使其包含-1到1范围内的像素，这是用于训练InceptionV3的图像格式。

# In[9]:


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# # 加载预训练的InceptionV3模型
# 现在，我们将创建一个tf.keras模型，其中输出层是InceptionV3体系结构中的最后一个卷积层。这层的输出形状是8x8x2048。使用最后一个卷积层，因为我们在此示例中使用了注意力。
# - 通过网络转发每个图像并将结果向量存储在字典中（image_name - > feature_vector）。
# - 在所有图像通过网络传递之后，挑选字典并将其保存到磁盘。

# In[11]:


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                               weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# ### 缓存从InceptionV3中提取的功能
# 将使用InceptionV3预处理每个映像并将输出缓存到磁盘。缓存RAM中的输出会更快但内存密集，每个映像需要8 * 8 * 2048个浮点数。在撰写本文时，这超出了Colab的内存限制（目前为12GB内存）。
# 
# 可以通过更复杂的缓存策略（例如，通过分割图像以减少随机访问磁盘I / O）来提高性能，但这需要更多代码。
# 
# 使用GPU在Colab中运行大约需要10分钟。如果您想查看进度条，可以：
# 
# 安装tqdm：
# 
# !pip install -q tqdm
# 
# 导入tqdm：
# 
# from tqdm import tqdm
# 
# 更改以下行：
# 
# for img, path in image_dataset:
# 
# 至：
# 
# for img, path in tqdm(image_dataset):

# In[13]:


get_ipython().system('pip install -q tqdm')
from tqdm import tqdm


# In[14]:


# 获取无重复图片
encode_train = sorted(set(img_name_vector))

# 获取训练图片特征，用于输入
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

# 读取图片，获取特征
for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features, 
                               (batch_features.shape[0],-1, batch_features.shape[3]))
    
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode('utf-8')
        np.save(path_of_feature, bf.numpy())


# ### 预处理并标记字幕
# - 首先，将对标题进行标记（例如，通过拆分空格）。这为我们提供了数据中所有独特单词的词汇表（例如，“冲浪”，“足球”等）。
# - 接下来，将词汇量限制为前5,000个单词（以节省内存）。将使用令牌“UNK”（未知）替换所有其他单词。
# - 然后，创建单词到索引和索引到单词的映射。
# - 最后，将所有序列填充为与最长序列相同的长度。

# In[16]:


# 寻找字幕最大长度
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# In[21]:


# 使用词频前5000的词构造词典
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                oov_token='<unk>',
                                                filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
# 获取tokenizer解析器
tokenizer.fit_on_texts(train_captions)
# 文本转token
train_seqs = tokenizer.texts_to_sequences(train_captions)


# In[22]:


# padding
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
# 最大长度
max_length = calc_max_length(train_seqs)


# ### 拆分训练集和测试集

# In[24]:


img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                               cap_vector,
                                                               test_size=0.2,
                                                               random_state=0)

len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)


# ### 创建用于培训的tf.data数据集
# 图片和标题已准备就绪！接下来，让我们创建一个tf.data数据集来用于训练模型。

# In[25]:


# 相关超参

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE
features_shape = 2048
attention_features_shape = 64


# In[26]:


# 导入图片特征
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap


# In[27]:


# 构建tf.data训练数据
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# 获取输入特征
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# 打乱分批次
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ### 模型
# 
# 模型架构的灵感来自Show，Attend和Tell论文。
# 
# 在这个例子中，从InceptionV3的下卷积层中提取特征，获得一个形状向量（8,8,2048）。
# 将它压成（64,2048）的形状。
# 然后，该向量通过CNN编码器（由单个完全连接的层组成）。
# RNN（此处为GRU）参与图像以预测下一个单词。

# In[28]:


# 注意力机制

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # 对hidden扩维 hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # 获取匹配得分score shape == (batch_size, 64, unit)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # 获取注意力权重
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, embedding_dim)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[29]:


# 编码cnn获得的图像特征
class CNN_Encoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


# In[38]:


# RNN解码器，解码出下一个词
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# In[39]:



# 网络模块


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


# In[40]:


# 优化器和损失函数
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# ### 模型保存

# In[41]:


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


# In[42]:


start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])


# ### 训练
# 您提取存储在相应.npy文件中的功能，然后通过编码器传递这些功能。
# 编码器输出，隐藏状态（初始化为0）和解码器输入（它是开始标记）被传递给解码器。
# 解码器返回预测和解码器隐藏状态。
# 然后将解码器隐藏状态传递回模型，并使用预测来计算损失。
# 使用教师强制决定解码器的下一个输入。
# 教师强制是将目标字作为下一个输入传递给解码器的技术。
# 最后一步是计算渐变并将其应用于优化器并反向传播。

# In[43]:


# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []


# In[44]:


@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


# In[45]:


EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
        ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# In[46]:


plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()


# ### 字幕！
# evaluate函数类似于训练循环，除了你不在这里使用教师强制。在每个时间步骤对解码器的输入是其先前的预测以及隐藏状态和编码器输出。
# 停止预测模型何时预测结束标记。
# 并存储每个时间步的注意力。

# In[47]:


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


# In[48]:


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


# In[49]:


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)
# opening the image
Image.open(img_name_val[rid])


# ### 尝试使用自己的图像
# 为了好玩，下面我们提供了一种方法，您可以使用我们刚刚训练过的模型为您自己的图像添加标题。请记住，它是在相对少量的数据上训练的，您的图像可能与训练数据不同（因此请为奇怪的结果做好准备！）

# In[52]:


image_url = 'https://tensorflow.org/images/surf.jpg'
image_extension = image_url[-4:]
image_path = tf.keras.utils.get_file('image'+image_extension,
                                     origin=image_url)

result, attention_plot = evaluate(image_path)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image_path, result, attention_plot)
# opening the image
Image.open(image_path)


# In[ ]:




