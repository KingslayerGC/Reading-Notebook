# %%
import numpy as np

# %%
## 获取数据
from sklearn.datasets import make_moons
X,y = make_moons(n_samples=500, noise=0.30, random_state=42)
# make_moons已实现数据分层平衡
from collections import Counter
Counter(y)
# 分割训练集和验证集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
## 集成学习：集成分类器
# 使用不同的分类器
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf = LogisticRegression(solver='liblinear', random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma='auto', random_state=42,
              probability=True) #这是为了让SVC可以输出预测概率，软投票必需
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf),
                                          ('svc', svm_clf)],voting='hard')
# 观察各分类器和集成分类器的准确性
from sklearn.metrics import accuracy_score
for clf in [log_clf, rnd_clf, svm_clf, voting_clf]:
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test,y_pred))


### 与硬投票不同，软投票可以让 sklearn 依据类概率来进行预测，预测概率为各分类器平均
### 因为给予高自信分类器更大的权重，软投票通常比硬投票表现得更好
### 把 voting超参数设置为 "soft" ，并确保所有子分类器都有预测概率的功能，
### 才能使用软投票，并使投票分类器可以输出预测概率   
    
# %%
## 集成学习：Bagging 和 Pasting （两者区别是采样后是否放回）
# Bagging示例：使用不同的训练集 
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=500, #50个基分类器
                            max_samples=100, #每个基分类器训练100个实例
                            bootstrap=True, #有放回
                            n_jobs=-1) #使用所有空闲cpu核
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
# 绘图对比普通决策树和 bagging分类器
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
# 下面定义了绘制边界的函数
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
# 绘图，可以看到 bagging 没有像普通决策树那样出现过拟合
plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
plt.show()

### 如果基分类器可以预测类别概率，那么BaggingClassifier会自动的运行软投票。
### Bagging结束时偏差比 Pasting更高，但减少了集合的方差，因此 Bagging 通常表现更好。

# %%
## out of bag
### 对于 Bagging 来说，一些实例可能被重复采样，而有些可能不会被采样。
### 这意味着对于每个基分类器，平均下来有37%的训练实例没有被采样。
### 这些样本就叫做 Out-of-Bag 实例。注意对于每一个分类器，它们的 37% 是不一样的。
### 因为在训练中分类器从来使用过 oob 实例，所以它可以在这些实例上进行评估
# 
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            bootstrap=True, n_jobs=-1,
                            oob_score=True) #注意设置obb分数可选
bag_clf.fit(X_train, y_train)
# 输出obb准确率
bag_clf.oob_score_
# 输出obb预测概率（前提是基分类器有预测功能）
bag_clf.oob_decision_function_

# %%
## BaggingClassifier 也支持采样特征
### 它被两个超参数 max_features 和 bootstrap_features 控制。
###他们的工作方式是对特征采样而不是实例采样，每一个分类器都会在随机的采样特征内进行训练。
### 当你在处理高维度输入时此方法尤其有效。
### 对训练实例和特征的采样被叫做随机贴片。
### 保留了所有的训练实例，但是对特征采样叫做随机子空间。
### 采样特征导致更多的预测多样性，用高偏差换低方差。

# %%
##随机森林
### 随机森林是决策树的一种集成，通常是通过 bagging 方法（有时是pasting 方法）
### 进行训练，通常把 max_samples 设置为训练集的大小。
# 随机森林实例
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
# 与上述随机森林基本等价的BaggingClassifier用法
bag_clf = BaggingClassifier(
        DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
        n_estimators=500, 
        max_samples=1.0, #浮点数表示随机采样比例，整数表示随机采样数 
        bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)

# %%
## 极端随机树
### 考虑到在每个节点上找到每个特征的最佳阈值是决策树最耗时的任务之一，我们可以
### 在每一个节点对每个特征随机指定一个阈值。
### 这种极端随机的树被简称为 Extremely Randomized Trees，或称 Extra-Tree。
### 这也是典型的用高偏差换低方差的方法。
### 自然地，Extra-Tree与规则的随机森林相比，训练速度更快。
### 很难分辨 ExtraTreesClassifier 和 RandomForestClassifier 到底哪个更好。
### 通常情况下，是通过交叉验证来比较它们（使用网格搜索调整超参数）。

# %%
## 特征重要度
### 重要的特征会出现在更靠近根部的位置。因此我们可以通过计算一个特征
### 在森林的全部树中出现的平均深度来预测特征的重要性。
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
# 输出所有特征的重要度
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

# %%
## 集成学习：Adaboost（适应性提升） 和 Gradient Boosting（梯度提升）
### Adaboost 使用的技术是使一个分类器对之前分类结果错误的训练实例多加关注，
### 以此修正之前的分类结果。
### Adaboost 算法的详细过程：先计算当前分类器的加权误差率（表示在对之前分类结果错误
### 的训练实例加权的条件下当前分类器的错误率），然后计算当前分类器的权重（表示当前分
### 类器的可信度），最后更新各实例的权重，被可信度高的当前分类器误判的实例权重会增加。
### 训练结束后，Adaboost 计算所有分类器的预测结果和对应的分类器权重，权重最高的结果
### 即为最终预测结果。
### sklearn 通常使用 Adaboost 的多分类版本 SAMME（在两类别情况下和 Adaboost等价）。
### SAMME.R使用预测概率而不是预测结果来计算分类器权重，迭代速率更快。
### 如果 Adaboost过拟合，可以尝试减少基分类器的数量或者对基分类器使用更强的正则化。
# Adaboost示例
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             n_estimators=200,
                             algorithm="SAMME.R", #仅基分类器有预测概率功能可用
                             learning_rate=0.5)
ada_clf.fit(X_train, y_train)
ada_clf.predict_proba([[1,1]])

# %%
## Gradient Boosting
### 梯度提升通过使用新分类器去拟合前面分类器预测的残差，来修正之前的分类结果。
# 生成回归数据
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X.ravel()**2 + 0.05 * np.random.randn(100)
# 自定义三层梯度提升回归
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)
y_pred1 = sum(tree.predict([[1]]) for tree in (tree_reg1, tree_reg2, tree_reg3))
# 调用GradientBoostingRegressor，与自定义等价
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,\
learning_rate=1.0) #决定每个树的贡献，较小的值需要更多的树
gbrt.fit(X, y)
y_pred2 = gbrt.predict([[1]])
# %%
## 以下代码试图寻找树的最佳数量
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_val, y_train, y_val = train_test_split(X, y)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)
# 每一阶段都返回一个误差值
errors = [mean_squared_error(y_val, y_pred) \
          for y_pred in gbrt.staged_predict(X_val)]
# 得到最佳树数目并应用
bst_n_estimators = np.argmin(errors)
gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)
# 换个思路，使用早停直接得到最佳树数目的回归
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
min_val_error = float("inf")
error_stable = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    # 计算每一阶段的误差
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_stable = 0
    else:
        error_stable += 1
    # 连续五次迭代误差没有改善则退出循环，视为找到最佳树数目
    if error_stable == 5:
        break

### GradientBoostingRegressor也支持指定用于训练每棵树的实例比例（即随机梯度提升）。
### 如果超参数 subsample=0.25 ，那么每个树都会在 25% 随机选择的训练实例上训练。
### 又一个高偏差换低方差的方法。

# %%
## 集成学习：Stacking
### Stacking的基本思想是：将训练集分成两个子集。第一个子集用来训练多个分类器，并对
### 第二个子集做出预测，将结果（特征）与真实值整合再训练一个分类器得到最终的预测结果。
### 实际应用中可以不止两层。
### sklearn中没有stacking功能。


























