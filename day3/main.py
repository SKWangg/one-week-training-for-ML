# 加载库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

# 读取数据
data_all = pd.read_csv('data_all.csv', encoding='gbk')

# 划分数据集
x = data_all.drop(columns=["status"]).as_matrix()
y = data_all[["status"]].as_matrix()
y = y.ravel()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)

# 归一化处理
scaler = StandardScaler()
scaler.fit(x_train)
x_train_standard = scaler.transform(x_train)
x_test_standard = scaler.transform(x_test)

# 定义评分函数
def get_scores(y_train, y_test, y_train_predict, y_test_predict, y_train_proba, y_test_proba):
    train_accuracy = metrics.accuracy_score(y_train, y_train_predict)
    test_accuracy = metrics.accuracy_score(y_test, y_test_predict)
    # 精准率
    train_precision = metrics.precision_score(y_train, y_train_predict)
    test_precision = metrics.precision_score(y_test, y_test_predict)
    # 召回率
    train_recall = metrics.recall_score(y_train, y_train_predict)
    test_recall = metrics.recall_score(y_test, y_test_predict)
    # F1-score
    train_f1_score = metrics.f1_score(y_train, y_train_predict)
    test_f1_score = metrics.f1_score(y_test, y_test_predict)
    # AUC
    train_auc = metrics.roc_auc_score(y_train, y_train_proba)
    test_auc = metrics.roc_auc_score(y_test, y_test_proba)
    # ROC
    train_fprs, train_tprs, train_thresholds = metrics.roc_curve(y_train, y_train_proba)
    test_fprs, test_tprs, test_thresholds = metrics.roc_curve(y_test, y_test_proba)
    plt.plot(train_fprs, train_tprs)
    plt.plot(test_fprs, test_tprs)
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()
    # 输出评分
    print("训练集准确率：", train_accuracy)
    print("测试集准确率：", test_accuracy)
    print("训练集精准率：", train_precision)
    print("测试集精准率：", test_precision)
    print("训练集召回率：", train_recall)
    print("测试集召回率：", test_recall)
    print("训练集F1-score：", train_f1_score)
    print("测试集F1-score：", test_f1_score)
    print("训练集AUC：", train_auc)
    print("测试集AUC：", test_auc)

# 逻辑回归
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=2018)
lr.fit(x_train_standard, y_train)
y_train_predict = lr.predict(x_train_standard)
y_test_predict = lr.predict(x_test_standard)
y_train_proba = lr.predict_proba(x_train_standard)[:, 1]
y_test_proba = lr.predict_proba(x_test_standard)[:, 1]
get_scores(y_train, y_test, y_train_predict, y_test_predict, y_train_proba, y_test_proba)

# SVM
from sklearn.svm import LinearSVC
svm_linearSVC = LinearSVC(random_state=2018)
svm_linearSVC.fit(x_train_standard, y_train)
y_train_predict = svm_linearSVC.predict(x_train_standard)
y_test_predict = svm_linearSVC.predict(x_test_standard)
y_train_proba = svm_linearSVC.decision_function(x_train_standard)
y_test_proba = svm_linearSVC.decision_function(x_test_standard)
get_scores(y_train, y_test, y_train_predict, y_test_predict, y_train_proba, y_test_proba)

# 决策树
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=2018)
tree.fit(x_train_standard, y_train)
y_train_predict = tree.predict(x_train_standard)
y_test_predict = tree.predict(x_test_standard)
y_train_proba = tree.predict_proba(x_train_standard)[:, 1]
y_test_proba = tree.predict_proba(x_test_standard)[:, 1]
get_scores(y_train, y_test, y_train_predict, y_test_predict, y_train_proba, y_test_proba)

# 随机森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=2018)
rf.fit(x_train_standard, y_train)
y_train_predict = rf.predict(x_train_standard)
y_test_predict = rf.predict(x_test_standard)
y_train_proba = rf.predict_proba(x_train_standard)[:, 1]
y_test_proba = rf.predict_proba(x_test_standard)[:, 1]
get_scores(y_train, y_test, y_train_predict, y_test_predict, y_train_proba, y_test_proba)

# GBDT
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=2018)
gb.fit(x_train_standard, y_train)
y_train_predict = gb.predict(x_train_standard)
y_test_predict = gb.predict(x_test_standard)
y_train_proba = gb.predict_proba(x_train_standard)[:, 1]
y_test_proba = gb.predict_proba(x_test_standard)[:, 1]
get_scores(y_train, y_test, y_train_predict, y_test_predict, y_train_proba, y_test_proba)

# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=2018)
xgb.fit(x_train_standard, y_train)
y_train_predict = xgb.predict(x_train_standard)
y_test_predict = xgb.predict(x_test_standard)
y_train_proba = xgb.predict_proba(x_train_standard)[:, 1]
y_test_proba = xgb.predict_proba(x_test_standard)[:, 1]
get_scores(y_train, y_test, y_train_predict, y_test_predict, y_train_proba, y_test_proba)

# LightGBM
from lightgbm import LGBMClassifier
lg = LGBMClassifier(random_state=2018)
lg.fit(x_train_standard, y_train)
y_train_predict = lg.predict(x_train_standard)
y_test_predict = lg.predict(x_test_standard)
y_train_proba = lg.predict_proba(x_train_standard)[:, 1]
y_test_proba = lg.predict_proba(x_test_standard)[:, 1]
get_scores(y_train, y_test, y_train_predict, y_test_predict, y_train_proba, y_test_proba)
