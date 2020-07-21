# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
## 正规方程
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# 正规方程计算
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# 绘图检视效果
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
plt.plot(X,y,'b.')
plt.plot(X_new,y_predict,'r-')
plt.show()
### 与sklearn的LinearRegression类效果完全一致
### 正规方程解法复杂度与样本数呈线性关系，与特征个数呈幂数关系

# %%
## 批量梯度下降
### 在梯度下降中，最重要的参数是步长，必须先进行特征放缩
eta = 0.1 #学习率
n_iterations = 1000 #迭代次数
m = 100 #样本数
theta = np.random.randn(2,1) #随机初始值
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
### 为了决定学习率，可以使用网格搜索；为了决定迭代次数，可以设置梯度变化下限

# %%
## 随机梯度下降
### 随机梯度下降，在每一步的梯度计算上只随机选取训练集中的一个样本
### 随机梯度下降可以跳过局部最优值，却不能达到最小值。解决办法是逐渐降低学习率
# 自定义随机梯度下降
n_epochs = 50 #迭代次数
m = 100 #样本数
t0, t1 = 5, 50 #learning_schedule的超参数
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2,1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
# 使用SGD回归完成随机梯度下降
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X,y.ravel())
sgd_reg.coef_
sgd_reg.intercept_

### 第三种迭代方式——小批量梯度下降，在迭代的每一步使用一个随机的小型样本集合

# %%
## 多项式回归
# 生成非线性数据
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
# 生成新特征
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2,include_bias=False)
X_poly = poly_features.fit_transform(X)
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_poly,y)

### PolynomialFeatures(degree=d)将包含所有特征的交叉组合

# %%
## 学习曲线
# 定义学习曲线绘制函数
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
# 简单线性回归绘制学习曲线
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
# 10次多项式回归绘制学习曲线
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((
("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
("sgd_reg", LinearRegression()),
))
plot_learning_curves(polynomial_regression, X, y)
plt.ylim(0,3)

# %%
## 正则化
np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
# 岭回归
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
# 随机梯度下降+Ridge 
sgd_reg = SGDRegressor(penalty="l2") 
sgd_reg.fit(X, y)
# lasso回归（倾向于把小参数直接设为0）
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
# 随机梯度下降+Lasso
sgd_reg = SGDRegressor(penalty="l1") 
sgd_reg.fit(X, y)
# 弹性网络回归（Ridge和Lasso的混合)
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
# 随机梯度下降+弹性网络
sgd_reg = SGDRegressor(penalty='elasticnet',l1_ratio=0.5) #Lasso正则项比率
sgd_reg.fit(X, y)

### SGDregressor正则项默认是l2,正则项系数默认是0.0001，Lasso正则项比率默认是0.15

# %%
## 早期停止法
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)
X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)
# 定义一个流水线，包括创建二次项和标准化
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler()),
        ])
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)
# 利用验证集误差完成早期停止的梯度下降
from sklearn.base import clone
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None,
                       learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_model = None
for epoch in range(1000):
    # warm_start=True使得每次梯度下降从上一次的终点开始
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val_predict, y_val)
    # 若验证误差下降，则视为找到更好模型；否则保留旧模型
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_model = clone(sgd_reg)

# %%
## 逻辑回归
from sklearn import datasets
iris = datasets.load_iris()
# 只选取一个特征
X = iris["data"][:, 3:]
# 改为二元变量
y = (iris["target"] == 2).astype(np.int)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)
# 输出最终预测
y_pred = log_reg.predict([[1.6]])
# 输出正例可能性
y_proba = log_reg.predict_proba([[1.6]])

# %%
## 逻辑回归的多输出
### 逻辑回归可以可以通过构造能整合多标签的得分函数和损失函数，
### 直接输出多分类,而无需使用 OvO，OvA
# 选两个特征，三种标签
X = iris["data"][:, (2, 3)]
y = iris["target"]
# Softmax 回归
softmax_reg = LogisticRegression(multi_class="multinomial",
                                 solver="lbfgs", C=10) #默认l2正则项
softmax_reg.fit(X, y)
# 输出所有情况可能性（本例三种）
softmax_reg.predict_proba([[5, 2]])

# %%
## 逻辑回归多输出分类图示
x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)
zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)
plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()

### 所有决策边界，即任何两类的边界都是直线

# %%













