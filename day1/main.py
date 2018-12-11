# -*- coding: utf-8 -*-
# 读取数据
import pandas as pd
data_all = pd.read_csv('data_all.csv',encoding='gbk')

# 划分数据集 （37分，随机种子2018）
from sklearn.model_selection import train_test_split
features = [x for x in data_all.columns if x not in ['status']]
x = data_all[features]
y = data_all['status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=2018)

# 构建逻辑回归模型
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state =2018)
lr.fit(x_train, y_train)

# 构建SVM模型
from sklearn.svm import LinearSVC
svm_linearSVC = LinearSVC(random_state=2018)
svm_linearSVC.fit(x_train, y_train)

# 构建决策树模型
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=2018)
tree.fit(x_train, y_train)

# 模型评价
lr_acc = lr.score(x_test, y_test)
svm_acc = svm_linearSVC.score(x_test, y_test)
tree_acc = tree.score(x_test, y_test)
print("LogisticRegression Acc: %f", lr_acc)
print("SVM Acc: %f", svm_acc)
print("DecisionTree Acc: %f", tree_acc)


#运行结果
LogisticRegression Acc: %f 0.7484232655921513
SVM Acc: %f 0.7484232655921513
DecisionTree Acc: %f 0.6846531184302733
