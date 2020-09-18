#!/usr/bin/env python
# coding: utf-8

# # PyTorch:数据的加载和预处理
# PyTorch通过torch.utils.data对一般常用的数据加载进行了封装，可以很容易地实现多线程数据预读和批量加载。
# 并且torchvision已经预先实现了常用图像数据集，包括前面使用过的CIFAR-10，ImageNet、COCO、MNIST、LSUN等数据集，可通过torchvision.datasets方便的调用

# ## Dataset类
# Dataset是一个抽象类，为了能够方便的读取，需要将要使用的数据包装为Dataset类。
# 自定义的Dataset需要继承它并且实现两个成员方法：
# 
# 1. `__getitem__()` 该方法定义用索引(`0` 到 `len(self)`)获取一条数据或一个样本
# 2. `__len__()` 该方法返回数据集的总长度

# In[5]:


from torch.utils.data import Dataset
import pandas as pd

#定义一个数据集
class BulldozerDataset(Dataset):
    def __init__(self, csv_file):
        self.df=pd.read_csv(csv_file)
    def __len__(self):
        # 返回df的长度
        return len(self.df)
    def __getitem__(self, idx):
        # 根据 idx 返回一行数据
        return self.df.iloc[idx].SalePrice


# In[7]:


ds_demo= BulldozerDataset('awewome-pypackages/pytorch/chapter2/median_benchmark.csv')

#实现了 __len__ 方法所以可以直接使用len获取数据总数
print(len(ds_demo))

#用索引可以直接访问对应的数据，对应 __getitem__ 方法
ds_demo[0]


# ## Dataloader
# 1. DataLoader为我们提供了对Dataset的读取操作，常用参数有：batch_size(每个batch的大小)、 shuffle(是否进行shuffle操作)、 num_workers(加载数据的时候使用几个子进程).
# 2. DataLoader返回的是一个可迭代对象，我们可以使用迭代器分次获取数据

# In[8]:


dl = torch.utils.data.DataLoader(ds_demo, batch_size=10, shuffle=True, num_workers=0)


# In[9]:


idata=iter(dl)
print(next(idata))


# In[11]:


# 常见的用法是使用for循环对其进行遍历

for i, data in enumerate(dl):
    print(i,data)
    break


# In[ ]:




