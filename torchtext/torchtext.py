#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import time
import random
import os


#     * API一览  torchtext.data *
#     torchtext.data.Example : 用来表示一个样本，数据+标签
#     torchtext.vocab.Vocab: 词汇表相关
#     torchtext.data.Datasets: 数据集类，__getitem__ 返回 Example实例
#     torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）
#     创建 Example时的 预处理
#     batch 时的一些处理操作。
#     torchtext.data.Iterator: 迭代器，用来生成 batch
#     torchtext.datasets: 包含了常见的数据集.

# ## Field对象

# * tokenize传入一个函数，表示如何将文本str变成token
# * sequential表示是否切分数据，如果数据已经是序列化的了而且是数字类型的，则应该传递参数use_vocab = False和sequential = False
# * Field类还允许用户指定特殊标记（用于标记词典外词语的unk_token，用于填充的pad_token，用于句子结尾的eos_token以及用于句子开头的可选的init_token）。设置将第一维是batch还是sequence（第一维默认是sequential），并选择是否允许在运行时决定序列长度还是预先就决定好

# In[2]:


from torchtext.data import Field

tokenize = lambda x: x.split()

TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)


# ## 构建Dataset

# > Fields知道怎么处理原始数据，现在我们需要告诉Fields去处理哪些数据。这就是我们需要用到Dataset的地方。Torchtext中有各种内置Dataset，用于处理常见的数据格式。 对于csv/tsv文件，TabularDataset类很方便。 以下是我们如何使用TabularDataset从csv文件读取数据的示例
# 

# In[3]:


from torchtext.data import TabularDataset
 
# 多标签分类
tv_datafields = [("id", None), # 我们不会需要id，所以我们传入的filed是None
                 ("comment_text", TEXT), ("toxic", LABEL),
                 ("severe_toxic", LABEL), ("threat", LABEL),
                 ("obscene", LABEL), ("insult", LABEL),
                 ("identity_hate", LABEL)]

# 读取训练集,验证集
trn, vld = TabularDataset.splits(
               path="data", # 数据存放的根目录
               train='train.csv', validation="valid.csv",
               format='csv',
               skip_header=True, # 如果你的csv有表头, 确保这个表头不会作为数据处理
               fields=tv_datafields)


# In[5]:


# 读取测试集
tst_datafields = [("id", None), # 我们不会需要id，所以我们传入的filed是None
                  ("comment_text", TEXT)]

# 单独读取一个文件
tst = TabularDataset(
           path="data/test.csv", # 文件路径
           format='csv',
           skip_header=True, # 如果你的csv有表头, 确保这个表头不会作为数据处理
           fields=tst_datafields)


# In[11]:


# 一次性读取三个文件

# 读取训练集,验证集,测试集
train, val, test = TabularDataset.splits(
        path='data', train='train.csv',
        validation='valid.csv', test='test.csv', format='csv',
        fields=[('Text', TEXT), ('Label', LABEL)])


# ## 词表

# > Torchtext将单词映射为整数，但必须告诉它应该处理的全部单词。 
# 
# > 在我们的例子中，我们可能只想在训练集上建立词汇表，所以我们运行代码：TEXT.build_vocab(trn)。这使得torchtext遍历训练集中的所有元素，检查TEXT字段的内容，并将其添加到其词汇表中。
# 
# > Torchtext有自己的Vocab类来处理词汇。Vocab类在stoi属性中包含从word到id的映射，并在其itos属性中包含反向映射。
# 

# In[12]:


TEXT.build_vocab(train)


# ## 构建迭代器

# > Iterators具有一些NLP特有的便捷功能。
# 
# * 对于验证集和训练集合使用BucketIterator.splits(),目的是自动进行shuffle和padding，并且为了训练效率期间，尽量把句子长度相似的shuffle在一起。
# * 对于测试集用Iterator，因为不用sort。
# * sort 是对全体数据按照升序顺序进行排序，而sort_within_batch仅仅对一个batch内部的数据进行排序。
# * sort_within_batch参数设置为True时，按照sort_key按降序对每个小批次内的数据进行降序排序。当你想对padded序列使用pack_padded_sequence转换为PackedSequence对象时，这是必需的。
# * 注意sort和shuffle默认只是对train=True字段进行的，但是train字段默认是True。所以测试集合可以这么写testIter = Iterator(tst, batch_size = 64, device =-1, train=False)写法等价于下面的一长串写法。
# * repeat 是否连续的训练无数个batch ,默认是False
# * device 可以是torch.device

# In[16]:


from torchtext.data import Iterator, BucketIterator
 
train_iter, val_iter = BucketIterator.splits((trn, vld), 
                                             # 我们把Iterator希望抽取的Dataset传递进去
                                             batch_sizes=(25, 25),
                                             device='-1', # 如果要用GPU，这里指定GPU的编号
                                             sort_key=lambda x: len(x.comment_text), # BucketIterator 依据什么对数据分组
                                             sort_within_batch=False,
                                             repeat=False   # repeat设置为False，因为我们想要包装这个迭代器层。
                                             )
         


# In[17]:


train_iter


# In[18]:


test_iter = Iterator(tst, batch_size=64, 
                     device='-1', 
                     sort=False, 
                     sort_within_batch=False, 
                     repeat=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




