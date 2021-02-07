
from sklearn.metrics import confusion_matrix



y_true = ['a','b','c','a']
y_pred = ['a','c','c','a']
labels = ['a','b','c']

confusion_matrix(y_true, y_pred)  # 行表示真是标签,列表示预测标签






# 分类报告
from sklearn.metrics import classification_report  
y_true = ['a','b','c','a']
y_pred = ['d','e','f','g']
print(classification_report(y_true,y_pred))
















