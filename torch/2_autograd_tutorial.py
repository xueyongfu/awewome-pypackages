#!/usr/bin/env python
# coding: utf-8

# 
# Autograd: 自动求导机制
# ===================================
# 
# PyTorch 中所有神经网络的核心是 ``autograd`` 包。
# 我们先简单介绍一下这个包，然后训练第一个简单的神经网络。
# 
# ``autograd``包为张量上的所有操作提供了自动求导。
# 它是一个在运行时定义的框架，这意味着反向传播是根据你的代码来确定如何运行，并且每次迭代可以是不同的。
# 
# 
# 示例
# 
# 张量（Tensor）
# --------
# 
# ``torch.Tensor``是这个包的核心类。如果设置
# ``.requires_grad`` 为 ``True``，那么将会追踪所有对于该张量的操作。 
# 当完成计算后通过调用 ``.backward()``，自动计算所有的梯度，
# 这个张量的所有梯度将会自动积累到 ``.grad`` 属性。
# 
# 要阻止张量跟踪历史记录，可以调用``.detach()``方法将其与计算历史记录分离，并禁止跟踪它将来的计算记录。
# 
# 为了防止跟踪历史记录（和使用内存），可以将代码块包装在``with torch.no_grad()：``中。
# 在评估模型时特别有用，因为模型可能具有`requires_grad = True`的可训练参数，但是我们不需要梯度计算。
# 
# 在自动梯度计算中还有另外一个重要的类``Function``.
# 
# 
# ``Tensor`` 和 ``Function``互相连接并生成一个非循环图，它表示和存储了完整的计算历史。
# 每个张量都有一个``.grad_fn``属性，这个属性引用了一个创建了``Tensor``的``Function``（除非这个张量是用户手动创建的，即，这个张量的
# ``grad_fn`` 是 ``None``）。
# 
# 如果需要计算导数，你可以在``Tensor``上调用``.backward()``。 
# 如果``Tensor``是一个标量（即它包含一个元素数据）则不需要为``backward()``指定任何参数，
# 但是如果它有更多的元素，你需要指定一个``gradient`` 参数来匹配张量的形状。
# 

# In[1]:


import torch


# In[10]:


# 创建一个张量并设置 requires_grad=True 用来追踪他的计算历史

x = torch.ones(2, 2, requires_grad=True)
print(x)

# 对张量进行操作:
y = x + 2
print(y)

# 结果y已经被计算出来了，所以，grad_fn已经被自动生成了。
print(y.grad_fn)

z = y * y * 3
print(z)

out = z.mean()
print(out)


# In[7]:


# .requires_grad_(...) 加个_,可以改变现有张量的 requires_grad属性. 

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)

a.requires_grad_(True)
print(a.requires_grad)

b = (a * a).sum()
print(b.grad_fn)


# 梯度
# ---------

# In[11]:


# 反向传播 因为 out是一个纯量（scalar），out.backward() 等于out.backward(torch.tensor(1))。

out.backward()

# 求导结果, print gradients d(out)/dx

print(x.grad)


# 求导过程:
# 
# 得到矩阵 ``4.5``.调用 ``out``
# *Tensor* “$o$”.
# 
# 得到 $o = \frac{1}{4}\sum_i z_i$,
# $z_i = 3(x_i+2)^2$ and $z_i\bigr\rvert_{x_i=1} = 27$.
# 
# 因此,
# $\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)$, hence
# $\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5$.
# 
# 

# In[45]:


# tensor.data和tensor.detach() 

# 1.都是变量从图中分离，但而这都是“原位操作
# 2.在 0.4.0 版本以前，.data 是用来取 Variable 中的 tensor 的，但是之后 Variable 被取消，.data 却留了下来。
# 3.现在我们调用 tensor.data，可以得到 tensor的数据 + requires_grad=False 的版本，而且二者共享储存空间，也就是如果修改其中一个，另一个也会变。因为 PyTorch 的自动求导系统不会追踪 tensor.data 的变化，所以使用它的话可能会导致求导结果出错。
# 4.官方建议使用 tensor.detach() 来替代它，二者作用相似，但是 detach 会被自动求导系统追踪，使用起来很安全 

# tensor.data
a = torch.tensor([1,2,3.], requires_grad = True)
out = a.sigmoid()
# 需要走注意的是，通过.data “分离”得到的的变量会和原来的变量共用同样的数据，而且新分离得到的张量是不可求导的，c发生了变化，原来的张量也会发生变化
c = out.data  
# 改变c的值，原来的out也会改变
c.zero_()     
print(c.requires_grad)
print(c)
print(out.requires_grad)
print(out)
print("-"*50)
# 对原来的out求导
out.sum().backward() 
# 不会报错，但是结果却并不正确
print(a.grad)  


# In[46]:


# tensor.detach()
 
a = torch.tensor([1,2,3.], requires_grad = True)
out = a.sigmoid()
# 需要走注意的是，通过.detach() “分离”得到的的变量会和原来的变量共用同样的数据，而且新分离得到的张量是不可求导的，c发生了变化，原来的张量也会发生变化
c = out.detach()  
# 改变c的值，原来的out也会改变
c.zero_()     
print(c.requires_grad)
print(c)
print(out.requires_grad)
print(out)
print("----------------------------------------------")
# 对原来的out求导
out.sum().backward() 
# 此时会报错，错误结果参考下面,显示梯度计算所需要的张量已经被“原位操作inplace”所更改了。
print(a.grad) 
