import pandas as pd
import os
import numpy as np 
from sklearn.model_selection import train_test_split


path = ''
DF = pd.read_excel(path,sheet_name=0)


df = DF[['', '']]
df.head(2)


df.groupby('').describe()


# 获取labels

df[''].unique()


# 文本去重, 空值处理, 字符串转化

df = df.drop_duplicates('', keep='last')

df.dropna(axis=0,  how='any', inplace=True)
df[''].fillna('缺失')

df[''] = df[''].astype('str')


# 数据清洗

def clean(line):
    
    return line


df[''] = df[''].apply(clean)


# 采样前进行训练集,测试集的划分

train_df,test_df = train_test_split(df, test_size=0.1, shuffle=True,random_state=42)


df_list = []

for name, sd in df.groupby(''):
    if len(sd) < 1000:
        df_list.append(pd.concat([sd, sd.sample(frac=0.6)]))
    elif 1000 < len(sd) < 2000:
        df_list.append(pd.concat([sd, sd.sample(frac=0.3)]))
    elif 3000 < len(sd) < 6000:
        df_list.append(sd.sample(n=3000))
    elif len(sd) > 6000:
        df_list.append(sd.sample(n=4500))

new_df = pd.concat(df_list)


new_df.groupby('').describe()


# 查看长度

new_df[''].apply(len).hist(bins=100)


# 训练集,验证集

train,dev = train_test_split(new_df, test_size=0.1, shuffle=True,random_state=42)


train.to_csv('data/train.tsv', sep='\t', index=False)
dev.to_csv('data/dev.tsv', sep='\t', index=False)
train_df.to_csv('data/train_all.tsv', sep='\t', index=False)
test_df.to_csv('data/test.tsv', sep='\t', index=False)
















