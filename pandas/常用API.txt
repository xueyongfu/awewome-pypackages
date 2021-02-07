import pandas as pd 

# 合并Series成DataFrame

a = pd.Series({'a':1, 'b':2, 'c':3})
b = pd.Series({'b':1, 'a':2, 'c':3})
c = pd.Series({'a':2, 'b':1, 'c':1})
pd.DataFrame([a,b,c])

# 取列

a = pd.DataFrame({'a':[1,2,3,4], 'b':[5,6,7,8]})

# 统计各个标签的数量

train_df['情感倾向'].value_counts().plot.bar()

# 数据集的基本预览

# 直接打印表对象
df

# 可以得到属性类别情况
df.info()

# 查看每列的类型
df.dtypes
 
# 查找多列，列索引必须要用中括号扩起来
complaints[['Complaint Type', 'Borough']][:10]
 
# 查找多行，这里的ix索引标签函数必须是中括号[]
student.ix[[0,2,4,5,7]]

#条件查询

# 查询出所有12岁以上的女生姓名、身高和体重
student[(student['Sex']=='F') & (student['Age']>12)][['Name','Height','Weight']]
 
# 查看某列中各个种类出现的次数
complaints['Complaint Type'].value_counts()
 
# 查看某列中属于某个种类的数据
# 为了得到噪音投诉，我们需要找到 Complaint Type  列为 Noise -Street/Sidewalk  的行。 我会告诉你如何做，然后解释发生了什么。
noise_complaints = complaints[complaints['Complaint Type'] == "Noise - Street/Sidewalk"]
 
# 将多个条件与 & 运算符组合
is_noise = complaints['Complaint Type'] == "Noise - Street/Sidewalk"
in_brooklyn = complaints['Borough'] == "BROOKLYN"
complaints[is_noise & in_brooklyn][:5]

# 函数中用来判断条件符合的数据集并返回
# df.query(条件式) 
df_new.query("duration > 100 & index == 'UK'")

# 保留指定标签范围内的数据

train_df[train_df['情感倾向'].isin(['0','1','-1'])]

# 日期的处理

train_df['time'] = pd.to_datetime('2020年' + train_df['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')
train_df['month'] =  train_df['time'].dt.month
train_df['day'] =  train_df['time'].dt.day
train_df['dayfromzero']  = (train_df['month']-1)*31 +  train_df['day']

# 排序sort()

# 将一列排序
Table[['某列']].sort()
 
# 将多列按照某一列排序
Table[['列1', '列2', '列3']].sort('列1')
 
# 值排序一般使用sort_values()
Table.sort_values(by = ['sex', 'age'])

# 抽样take()

# 对于一个dataframe，take函数可以根据sampler按照行索引抽取数据
df.take(sampler)

# 哑变量get_dummies()

pd.get_dummies(df['key']，prefix='key')

# 缺失值

df.dropna()

df.fillna()

df.isnull()   df.isna()  #isnull是isna的别名

# 分组

grp1=df1.groupby('symbol')

#根据两列数据将数据分组
grp2=df1.groupby(['symbol','tdate'])

#通过组名访问 组名对应的数据
print(grp1.get_group('001'))
print(grp2.get_group(('001','201901')))

# 表连接

# merge()函数

# 1.默认情况下，merge函数实现的是两个表之间的内连接，即返回两张表中共同部分的数据。
# 2.可以通过how参数设置连接的方式，inner或空为内连接，left为左连接；right为右连接；outer为外连接。
# 3.内连接至显示有共同索引的，outer是全部显示

# on=用来指定连接轴,当两张表中的连接轴列名不同时,通过left_on, right_on进行连接
pd.merge(student, score, on='name', how=right)
# pd.merge(student, score, left_on='name',right_on='名字' how=right)
pd.merge(student, score, left_on='key', right_index=True)

# suffixes=(‘_x’,’_y’) 指的是当左右对象中存在除连接键外的同名列时，结果集中的区分方式，可以各加一个小尾巴。
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))



# 表连接 join函数
# join默认是以索引连接，内连接

# 设置外连接
left2.join(right2, how='outer')

# 指定连接键
left1.join(right1, on='key')

# 只要right2, another中的索引有一个可以与left的索引匹配，就可以建立连接
left2.join([right2, another])


# 统计分析函数

# 一般统计特征函数
d1.count() #非空元素计算
d1.min() #最小值
d1.max() #最大值
d1.idxmin() #最小值的位置
d1.idxmax() #最大值的位置
d1.quantile(0.1) #10%分位数
d1.sum() #求和
d1.mean() #均值
d1.median() #中位数
d1.mode() #众数
d1.var() #方差
d1.std() #标准差
d1.mad() #平均绝对偏差
d1.skew() #偏度
d1.kurt() #峰度
d1.describe() #一次性输出多个描述性统计指标

# 关于相关系数的计算可以调用pearson方法、kendell方法、spearman方法，默认使用pearson方法。计算的是任意两列的相关系数。
df.corr()

# 如果只想关注某一个变量与其余变量的相关系数的话，可以使用corrwith,如下方只关心x1与其余变量的相关系数:
df.corrwith(df['x1'])

# 数值型变量间的协方差矩阵
df.cov()

# 累计统计特征函数
# 使用格式
pd.rolling_mean(D, k)  #意思是每k个数计算一次均值。

# 函数使用:apply, agg,transform

# agg()

# 1.agg()和apply()区别
# agg函数内调用的函数只能对分组进行聚合使用，比如mean,sum。apply的应用更广泛，apply函数可以说是它的泛化，比如你可以用apply实现组内排序，但是#
# agg函数并不能。
 
# 2.agg()使用的多种形式
grouped_pct.agg(['mean', 'std', peak_to_peak])
 
#给计算结果一个别名
grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])  
 
#列表形式
functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
 
#元组形式
ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
grouped['tip_pct', 'total_bill'].agg(ftuples)
 
#字典形式
grouped.agg({'tip' : np.max, 'size' : 'sum'})
 
grouped.agg({'tip_pct' : ['min', 'max', 'mean',  'std'], 'size' : 'sum'})  #混合形式


# apply()

# apply()作用与数据的每一列，或者每一行

def stats(x):
    return pd.Series([x.count(),x.min(),x.idxmin()],index = ['Count','Min','Whicn_Min'])
#将df数据框的每一列应用status函数,默认axis=0
df.apply(stats, axis=0)


# transform()

# 一般函数性质
df.transform(lambda x: (x - x.mean()) / x.std())

# 特别函数性质
# 在groupby之后使用aggregate，transform则不对数据进行聚合，它会在对应行的位置生成聚合函数值。


# 缺失值处理

# 直接删除
# 默认对行进行操作,会删除任意含有缺失值的行
df.dropna()      

# 删除指定行
student.drop([1,4,7])

# 删除指定列
Student.drop(['列1','列2'], axis=1)

#只删除全是缺失值的行
df.drop(how=all) 

# 填补法

# 1.使用0填充
df.fillna(0)

# 2.前向填充和后向填充
df.fillna(method=ffill)
df.fillna(method=bfill)

# 3.常量填充, 或者均值填充，中位数填充
df.fillna({'列1': 2，'列2': 3})

# 插值法
# 自定义列向量拉格朗日插值函数

#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
from scipy.interpolate import lagrange #导入拉格朗日插值函数
def ployinterp_column(s, n, k=5):
    y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] #取数
    y = y[y.notnull()] #剔除空值
    return lagrange(y.index, list(y))  #插值并返回插值结果
 
#逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        if (data[i][j]).isnull(): #如果为空即插值。
            data[i][j] = ployinterp_column(data[i], j)

# 数据规范化

#最小-最大规范化，会作用于每一列
(data - data.min())/(data.max() - data.min())
 
#零-均值规范化，会作用于每一列
(data - data.mean())/data.std()

# 连续属性离散化

# 将数据分配到一个数据空间中
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)   #由bins可以得到四个空间，然后将ages中每一个数字放入合适的空间中
 
# 获得每个数据的空间名
cats.labels
 
# 获得总共划分了多少空间
cats.levels
 
# 查看每个空间有多少数据被划分进去
pd.value_counts(cats)
 
# 指定每个空间的开闭口的方位
pd.cut(ages, [18, 26, 36, 61, 100], right=False)
 
# 为空间命名
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)



# 默认qcut()将数据n等分，可以同给给出累计分布值的方式对数据进行划分
data = np.random.randn(100)
 
cats = pd.qcut(data, 4) # Cut into quartiles
print(pd.value_counts(cats))
 
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1])  #按照累计分布直进行数据切分












