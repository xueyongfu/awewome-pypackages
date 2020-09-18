#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 神经网络例子

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1表示input image channel, 6表示output channels, 5表示是5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)


# In[11]:


# 在模型中必须要定义 forward 函数，backward 函数（用来计算梯度）会被autograd自动创建。 
# 可以在 forward 函数中使用任何针对 Tensor 的操作。
# net.parameters()返回可被学习的参数（权重）列表和值
# 输出的单一维度的是偏差b

params = list(net.parameters())
print('lengths:',len(params))
for i in range(len(params)):
    print(params[i].size())


# In[14]:


# 添加损失函数

output = net(input)
target = torch.randn(10)  # 随机值作为样例
target = target.view(1, -1)  # 使target和output的shape相同

criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)


# In[15]:


# 现在，如果在反向过程中跟随loss ， 使用它的 .grad_fn 属性，将看到如下所示的计算图。
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
# 所以，当我们调用 loss.backward()时,整张计算图都会 根据loss进行微分，而且图中所有设置为requires_grad=True的张量 将会拥有一个随着梯度累积的.grad 张量。

# 为了说明，让我们向后退几步:

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# 反向传播
# --------
# 调用loss.backward()获得反向传播的误差。
# 
# 但是在调用前需要清除已存在的梯度，否则梯度将被累加到已存在的梯度。
# 
# 现在，我们将调用loss.backward()，并查看conv1层的偏差（bias）项在反向传播前后的梯度。
# 
# 
# 

# In[16]:


# 清除梯度
net.zero_grad()     

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# 更新权重
# ------------------
# 在实践中最简单的权重更新规则是随机梯度下降（SGD）：
# 
#      ``weight = weight - learning_rate * gradient``
# 
# 我们可以使用简单的Python代码实现这个规则：
# 
# ```python
# 
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
# ```

# In[11]:


import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()  
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()   


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




