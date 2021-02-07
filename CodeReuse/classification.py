import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('a.csv')

# 多列合并成一个标签
df.fillna(value='未知', inplace=True)
df['label'] = df['是否合理'].str.cat(
    [df['解决情况'], df['具体理由/未解决细分'], df['具体理由']], sep='-')

# 生成标签的统计数据
df['label'].value_counts().to_csv('train_data/dis.csv')

# 数据清洗
def clean(line):
    return re.sub('\s', '', str(line)[:512])
df['反馈结果'] = df['反馈结果'].apply(clean)


# 查看长度
plt.figure()
df['反馈结果'].apply(len).hist(bins=100)
plt.show()


# 训练集,验证集
train, dev = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)


# 数据采样
train_list = []
for name, sd in train.groupby('label'):
    if len(sd) < 30:
        train_list.append(
            pd.concat([sd, sd.sample(n=30 - len(sd), replace=True)]))
    elif len(sd) < 50:
        train_list.append(
            pd.concat([sd, sd.sample(n=50 - len(sd), replace=True)]))
    else:
        train_list.append(sd)
new_train = pd.concat(train_list)

# 生成标签的统计数据
new_train.groupby('label').describe().to_csv('train_data/dis2.csv')

new_train.to_csv('train_data/train.tsv', sep='\t', index=False)
dev.to_csv('train_data/dev.tsv', sep='\t', index=False)
df.sample(frac=1).to_csv('train_data/train_all.tsv', sep='\t', index=False)

# 权重调整
train['label'].value_counts()
weights = 1 / (train['label'].value_counts() / (train['label'].value_counts().max()))
print(weights.to_list())
print(weights.index)
