
from __future__ import print_function
import torch



# 创建一个 5x3 矩阵, 但是未初始化:

x = torch.empty(5, 3)
print(x)




# 创建一个随机初始化的矩阵:

x = torch.rand(5, 3)
print(x)




# 创建一个0填充的矩阵，数据类型为long:

x = torch.zeros(5, 3, dtype=torch.long)
print(x)




# 创建tensor并使用现有数据初始化:

x = torch.tensor([5.5, 3])
print(x)




# 根据现有的张量创建张量。 这些方法将重用输入张量的属性，例如， dtype，除非设置新的值进行覆盖

x = x.new_ones(5, 3, dtype=torch.double)      # new_* 方法来创建对象
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype!
print(x)                                      #  对象的size 是相同的，只是值和类型发生了变化




# 获取 size
# 译者注：使用size方法与Numpy的shape属性返回的相同，张量也支持shape属性，后面会详细介绍

print(x.size())




# 加法1:

y = torch.rand(5, 3)
print(x + y)


# 加法2:
print(torch.add(x, y))

# 提供输出tensor作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)




# 任何以``_``结尾的操作都会用结果替换原变量.
# 例如: ``x.copy_(y)``, ``x.t_()``, 都会改变 ``x``

m = torch.tensor([1,1,1])
n = torch.tensor([2,2,2])

m.copy_(n)   #相当于m=n
print(m)

m.add_(n)   #相当于m=m+n
print(m)




# 你可以使用与NumPy索引方式相同的操作, 来进行对张量的操作
# torch张量的切片操作

print(x[:, 1])




# 改变张量的维度:view()

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) 
print(x.size(), y.size(), z.size())




# 如果你有只有一个元素的张量，使用.item()来得到Python数据类型的数值

x = torch.randn(1)
print(x)
print(x.item())




# NumPy 转换
# 将一个Torch Tensor转换为NumPy数组是一件轻松的事，反之亦然。Torch Tensor与NumPy数组共享底层内存地址，修改一个会导致另一个的变化。


# 将一个Torch Tensor转换为NumPy数组
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# 张量和numpy数据同时变换
a.add_(1)
print(a)
print(b)


#  NumPy Array 转化成 Torch Tensor
# 
# 使用from_numpy自动转化
# 
# 



# NumPy Array 转化成 Torch Tensor, 使用from_numpy自动转化

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# 所有的 Tensor 类型默认都是基于CPU， CharTensor 类型不支持到 NumPy 的转换.



# CUDA 张量
# 使用.to 方法 可以将Tensor移动到任何设备中

# is_available 函数判断是否有cuda可以使用, `torch.device`用来设置使用的cuda设备
if torch.cuda.is_available():
    device = torch.device("cuda")          
    y = torch.ones_like(x, device=device)  # 1.直接从GPU创建张量
    x = x.to(device)                       # 2.或者直接使用``.to("cuda")``将张量移动到cuda中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 也会对变量的类型做更改

