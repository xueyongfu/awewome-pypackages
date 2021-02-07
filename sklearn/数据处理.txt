import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs

# 训练集,验证集,测试集的划分

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(train_df, test_size=0.15, shuffle=True, random_state=4)
train_df, dev_df = train_test_split(train_df, test_size=0.15, shuffle=True, random_state=4)

# 混淆矩阵

from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

# 行表示true_label, 列表示predict_label
confusion_matrix(y_true, y_pred)

from sklearn.metrics import roc_auc_score

y_scores=np.array([  0, 0, 0, 0, 0, 1, 0])
y_true=  np.array([  0, 0, 0, 0, 1, 1, 0])

roc_auc_score(y_true, y_scores)

# 分类结果报告

from sklearn.metrics import classification_report
y_true = ['d','a', 'b', 'c']
y_pred = ['c', 'a', 'c', 'c']

# y_true或者y_pred是数字时,target_names按照数据大小生序对应, 如果是字符串,按照首字母对应
target_names = ['class 0', 'class 1', 'class 2', 'class_3']
print(classification_report(y_true, y_pred, target_names=target_names))
















