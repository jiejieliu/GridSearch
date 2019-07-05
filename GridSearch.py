from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

iris = load_iris()

# 只采用网格搜索，最终的表现好坏与初始数据的划分结果有很大的关系，需要采用交叉验证的方式来减少偶然性。
# 交叉验证经常与网格搜索进行结合，是参数评价的一种方法
best_score = 0.0
# 把要调整的参数以及其候选值 列出来；
param_grid = {"gamma": [0.001, 0.01, 0.1, 1, 10, 100],
              "C": [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameters:{}".format(param_grid))

grid_search = GridSearchCV(SVC(), param_grid, cv=5)  # 实例化一个GridSearchCV类
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=10)
grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
print("Best parameters:{}".format(grid_search.best_params_))
print("Best score on train set:{:.2f}".format(grid_search.best_score_))
