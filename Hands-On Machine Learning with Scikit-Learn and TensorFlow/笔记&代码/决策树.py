# %%
import numpy as np

# %%
## 决策树
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X = iris.data[:, 2:]
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
# 将对应叶节点的各类比例作为预测概率输出
tree_clf.predict_proba([[5, 1.5]])
# 将比例最大的一类作为预测种类输出
tree_clf.predict([[5, 1.5]])

### Scikit-Learn 用的是 CART 算法（仅产生二叉树）。CART是一种贪婪算法，通过寻求合
### 适的分割特征和分割阈值，使每一步分割的代价函数最小化，以此获得一个非最佳解。
### 决策树不需要太多的数据预处理，尤其是不需要进行特征缩放或者归一化。
### 决策树可以采用Gini或熵来衡量不纯度，其中Gini倾向于分出数目最大的类，熵则较为平衡
### 代价函数：m1/m*G1+m2/m*G2
### Gini指数：G=1-(m1/m)^2-(m2/m)^2    熵：G=-(m1/m)*log(m1/m)-(m2/m)*log(m2/m)
### 对于非参数模型，容易出现过拟合的情况，因此通常需要设置一些正则超参数

# %%
## 决策树可视化
# 定义正则化程度不同的决策树
from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)
deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf1.fit(Xm, ym)
deep_tree_clf2.fit(Xm, ym)
# 定义边界绘制函数
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)
# 对比绘制决策树图
plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=16)
plt.subplot(122)
plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
plt.show()

# %%
## 决策树回归
# 构建数据
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2 + np.random.randn(m, 1) / 10
# 运用决策树回归
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
# 将对应叶节点所有样本的平均值作为预测值输出
tree_reg.predict([[0.5]])

### 与决策树分类类似，决策树回归用mse取代原代价函数中的G，使用贪婪算法执行分类
### 决策树回归使用样本平均值进行预测，因此预测函数为分段函数
### 同样地，决策树回归需设置一些正则超参数来防止过拟合

# %%
### 决策树倾向于设定正交化的决策边界（所有边界都是和某一个轴相垂直的），
### 这使得它对训练数据集的旋转很敏感，一种解决方法是使用 PCA 主成分分析。
### 另外，决策树对数据的微小变化也比较敏感，而随机森林能通过多棵树的平均进行改善










