# 1.条形图
# 条形图是用条形的长度表示各类别频数的多少，其宽度（表示类别）则是固定的
import matplotlib.pyplot as plt
%matplotlib inline

plt.bar([1, 3, 5, 7, 9], [5, 4, 8, 12, 7], label='graph 1')
plt.bar([2, 4, 6, 8, 10], [4, 6, 8, 13, 15], label='graph 2')

plt.legend()

plt.xlabel('number')
plt.ylabel('value')

plt.show()

# 2.直方图
# 直方图是用面积表示各组频数的多少，矩形的高度表示每一组的频数或频率，宽度则表示各组的组距，因此其高度与宽度均有意义
import matplotlib.pyplot as plt
%matplotlib inline

salary = [2500, 3300, 2700, 5600, 6700, 5400, 3100, 3500, 7600, 7800, 
          8700, 9800, 10400,12,1,567,7887,67]

group = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]

#可以使用两种方式尽心分组
plt.hist(salary,bins=group, histtype='bar', rwidth=0.8)
plt.hist(salary,bins=20, histtype='bar', rwidth=0.8)

plt.legend()

plt.xlabel('salary-group')
plt.ylabel('salary')

plt.show()

import numpy as np
 
import matplotlib
matplotlib.use('Agg')
 
from matplotlib.pyplot import plot,savefig
 
x=np.linspace(-4,4,30)
y=np.sin(x);
 
plot(x,y,'--*b')
 
savefig('./MyFig.png')
















