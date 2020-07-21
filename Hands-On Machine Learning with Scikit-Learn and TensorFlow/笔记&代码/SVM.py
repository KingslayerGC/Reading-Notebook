# %%
import numpy as np

# %%
## SVM线性分类器
from sklearn import datasets
iris = datasets.load_iris()
# 选两个特征
X = iris["data"][:, (2, 3)]
# 改为二分类
y = (iris["target"] == 2).astype(np.float64)
# 使用LinearSVC分类
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
svm_clf = Pipeline((
("scaler", StandardScaler()),
("linear_svc", LinearSVC(C=1, loss="hinge")),
))
svm_clf.fit(X, y)
# SVM无法输出正例概率
svm_clf.predict([[3.2, 1.1]])

### 惩罚参数越大，分隔边界越宽；样本量大于特征数时，dual建议设为False

# %%
## 其他使用线性SVM的方法
# 在SVC类中选择线性核
from sklearn.svm import SVC
svm_clf = SVC(kernel="linear", C=1) #速度较慢，不推荐
# 在SGD类(即随机梯度下降)中选择hinge损失函数
from sklearn.linear_model import SGDClassifier
svm_clf = SGDClassifier(loss='hinge') #适合处理较大的数据集

# %%
## 非线性SVM
# 将特征取交叉多元，再使用线性SVM
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
X,y = make_moons(n_samples=100, noise=0.15, random_state=42)
poly_svm_clf = Pipeline([
        ('polynomial', PolynomialFeatures(degree=3)),
        ('standard', StandardScaler()),
        ('linearsvc', LinearSVC(C=10, loss="hinge"))
        ])
poly_svm_clf.fit(X,y)
# 定义画图函数
import matplotlib.pyplot as plt
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    # 叠加颜色来体现边界
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
# 展示分类器效果
plot_predictions(poly_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
# 运用“核技巧”可以获得和高次多项式一样好的结果。但不增加任何特征，不会导致模型变慢
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5)) #coef参数决定高次项的比例
))
poly_kernel_svm_clf.fit(X, y)

# %%
## 另一种方法是增加相似特征
### 运用高斯径向基函数(RBF)可以生成新特征（又叫相似特征）
# 类似地，运用高斯核可以获得和相似特征法一样好的结果，而不花费大量的计算成本
rbf_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
))
rbf_kernel_svm_clf.fit(X, y)

### gamma参数越大，钟形曲线越窄，每个样本的影响范围减少，拟合越强（即效果与C一致）

# %%
## SVM回归
### 不再试图在两个类别之间找到尽可能大的“街道”（即间隔），而是限制间隔违规情况下，
### 尽量放置更多的样本在“街道”上。“街道”的宽度由超参数 ϵ 控制。
# 使用LinearSVR完成SVM回归
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
# 使用多项式核回归
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)

# %%




































