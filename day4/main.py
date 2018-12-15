# 加载库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


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

# 网格搜索五折交叉验证
from sklearn.model_selection import GridSearchCV
def gridsearch(model, parameters):
    grid = GridSearchCV(model, parameters, scoring='accuracy', cv=5)
    grid = grid.fit(x_train_standard, y_train)
    if hasattr(model, 'decision_function'):
        y_predict_proba = grid.decision_function(x_test_standard)
    else:
        y_predict_proba = grid.predict_proba(x_test_standard)[:, 1]
    print('best score:', grid.best_score_)
    print(grid.best_params_)
    print('test score:', grid.score(x_test_standard, y_test))
    print('AUC:', metrics.roc_auc_score(y_test, y_predict_proba))

# 逻辑回归
from sklearn.linear_model import LogisticRegression
parameters = {'C': [0.01, 0.1, 1, 10, 100]}
lr = LogisticRegression(random_state=2018)
gridsearch(lr, parameters)

# SVM
from sklearn.svm import LinearSVC
parameters = {'C': [0.01, 0.1, 1, 10, 100]}
svm_linearSVC = LinearSVC(random_state=2018)
gridsearch(svm_linearSVC, parameters)

# 决策树
from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [0.01, 0.1, 1, 10, 100]}
tree = DecisionTreeClassifier(random_state=2018)
gridsearch(tree, parameters)


# 随机森林
from sklearn.ensemble import RandomForestClassifier
parameters = {'max_depth': [0.01, 0.1, 1, 10, 100], 'n_estimators': range(5,50,5)}
rf = RandomForestClassifier(oob_score=True, random_state=2018)
gridsearch(rf, parameters)

# GBDT
from sklearn.ensemble import GradientBoostingClassifier
parameters = {'max_features': [0.01, 0.1, 1], 'n_estimators': range(5,50,5)}
gb = GradientBoostingClassifier(random_state=2018)
gridsearch(gb, parameters)

# XGBoost
from xgboost import XGBClassifier
parameters = {'gamma': [0.01, 0.1, 1, 10, 100], 'min-child-weight': [0.01, 0.1, 1, 10, 100]}
xgb = XGBClassifier(random_state=2018)
gridsearch(xgb, parameters)

# LightGBM
from lightgbm import LGBMClassifier
parameters = {'max_depth': [1, 10, 20], 'n_estimators': [ 1, 10, 20]}
lg = LGBMClassifier(random_state=2018)
gridsearch(lg, parameters)
