# %%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
## 加载数据
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"].astype(int)
# MINST数据为一个70000×784矩阵，每一张图片为28×28像素，共有70000张图片

# %%
## 查看数据
# 显示第36001张图片
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
# 查看对应标签（即真实数字）
y[36000]

# %%
## 分割训练集
# MNIST数据集已被分成一个训练集（前 60000 张图片）和一个测试集
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# 打乱训练集
shuffle_index = np.random.permutation(60000) #得出一种可能的排列
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# %%
## 先训练一个简单的二分类器
# 将所有数据分为是5或非5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# 随机梯度下降分类器SGD
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# 预测示例
sgd_clf.predict([some_digit]) #需先将一维数组转为二维

# %%
## 交叉验证准确性
# 自定义的交叉检验
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
# 将数据分为3折
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf) #克隆分类器
    #对每折进行检验
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    # 计算准确度并打印
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
# 交叉验证，结果与自定义完全一致
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

### 受偏斜类的影响，这里得出的精度非常高，但并不能代表分类器的真实性能

# %%
## 精确度指标
# 计算混淆矩阵，输出每一类别被分类成其他类别的次数
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
sgd_y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confuse_mat = confusion_matrix(y_train_5, sgd_y_train_pred)
from sklearn.metrics import precision_score, recall_score, f1_score
# 计算准确率（ precision=TP/TP+FP ）
sgd_precision = precision_score(y_train_5, sgd_y_train_pred)
# 计算召回率（ recall=TP/TP+FN ）
sgd_recall = recall_score(y_train_5, sgd_y_train_pred) 
# 计算F1值（ F1 = 2×precision×recall/(precision+recall) ）
sgd_f1 = f1_score(y_train_5, sgd_y_train_pred)


# %%
## 调整决策阈值——根据准确率或召回率
# 查看一个例子的SGD决策分数（分数高即判断阳性）
sgd_y_score = sgd_clf.decision_function([some_digit])
# 查看所有分数
sgd_y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
method="decision_function")
# 计算每一个阈值对应的准确率和召回率
from sklearn.metrics import precision_recall_curve
sgd_precisions, sgd_recalls, thresholds =\
precision_recall_curve(y_train_5, sgd_y_scores)
# 绘图
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
def plot_pr_curve(precision, recall, label=None):
    plt.plot(precision, recall, linewidth=2, label=label)
    plt.plot([0, 1], [1, 0], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
plot_precision_recall_vs_threshold(sgd_precisions, sgd_recalls, thresholds)
plt.show()
plot_pr_curve(sgd_precisions, sgd_recalls)
plt.show()
# 应用一个更高的阈值
y_train_pred_90 = (sgd_y_scores > 70000)

# %%
## 调整决策阈值——根据ROC曲线
# 计算每一个阈值对应的真阳性率（ TP/(TP+FN) ）和假阳性率（ FP/(TN+FP) ）
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, sgd_y_scores)
# 绘制ROC曲线
def plot_roc_curve(fpr, tpr, linetype='b', label=None):
    plt.plot(fpr, tpr, linetype, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1],'r')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()
# 计算AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, sgd_y_scores)

### 一个笨拙的规则是，当正例很少，或者当你关注假阳性多于假阴性的时候,
### 优先使用 PR 曲线，其他情况使用 ROC 曲线。

# %%
## 训练随机森林分类器
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
# 得到所有预测概率（决策分数和阳性概率，分类器常居其一）
forest_y_probas = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
method="predict_proba")

# %%
## 绘制两个分类器的ROC曲线
# 将阳性概率视为决策分数
forest_y_scores = forest_y_probas[:, 1]
# 计算每一个阈值对应的真阳性率和假阳性率
forest_fpr, forest_tpr, thresholds = roc_curve(y_train_5,forest_y_scores)
plot_roc_curve(fpr, tpr, 'b:', label="SGD")
plt.plot(forest_fpr, forest_tpr, label="Random Forest")
plt.legend(loc="best")
plt.show()

# %%
## 计算随机森林分类器指标
# AUC
from sklearn.metrics import roc_auc_score
forest_auc = roc_auc_score(y_train_5, forest_y_scores)
from sklearn.model_selection import cross_val_predict
forest_y_train_pred = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
from sklearn.metrics import precision_score, recall_score, f1_score
# precison
forest_precision = precision_score(y_train_5, forest_y_train_pred)
# recall
forest_recall = recall_score(y_train_5, forest_y_train_pred) 
# f1
forest_f1 = f1_score(y_train_5, forest_y_train_pred)

# %%
## 多分类器
### 一些算法（比如随机森林分类器或者朴素贝叶斯分类器）可以直接处理多类分类问题。
### 其他一些算法（比如 SVM 分类器或者线性分类器）则是严格的二分类器。
### 对于多分类问题，将二分类扩展成多分类通常有 OvA 和 OvO 两种思路。
### 对于一些在训练集的大小上很难扩展的算法（比如 SVM ），OvO 是比较好的，因为它
### 在小的数据集上面可以更多地训练。但是对于大部分二分类器来说，OvA 是更好的选择。
### skLearn 可以探测出你想使用一个二分类器去完成多分类的任务，并自动地执行 OvA（SVM 分类器除外）
### 可以强制 skLearn 使用 OvO 策略或者 OvA 策略
# skLearn 实际上训练了 10 个二分类器，计算出了10个决策数值，选择数值最高的那个类
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
# 返回的是 10 个数值，一个数值对应于一个类
some_digit_scores = sgd_clf.decision_function([some_digit])
# 强制使用 OvO 策略
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
# 形如随机森林的分类器自带多分类功能
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
# 得到所有类别概率
forest_clf.predict_proba([some_digit])
# 交叉检验
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# %%
## 误差分析
# 数据标准化，模型表现更佳
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sta = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_sta, y_train, cv=3, scoring="accuracy")
# 计算混淆矩阵
y_train_pred = cross_val_predict(sgd_clf, X_train_sta, y_train, cv=3)
confuse_mat = confusion_matrix(y_train, y_train_pred)
# 比较错误率而不是错误数
row_sums = confuse_mat.sum(axis=1, keepdims=True)
norm_confuse_mat = confuse_mat / row_sums
# 排除对角线的干扰
np.fill_diagonal(norm_confuse_mat, 0)
plt.matshow(norm_confuse_mat, cmap=plt.cm.gray)
plt.show()

# %%
# 深入分析3和5混淆原因(画图函数没看懂)
def plot_digits(instances, images_per_row=10):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image,cmap = mpl.cm.binary)
    plt.axis("off")
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()

# %%
## 二元多标签分类
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
# 创建二标签数组
y_multilabel = np.c_[y_train_large, y_train_odd]
# KNeighborsClassifier支持多标签分类，但不是所有分类器都可以
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
# 输出对两种标签的预测
knn_clf.predict([some_digit])
#  使用交叉验证检验多标签准确性，需要花费很多时间
#y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
#f1_score(y_train, y_train_knn_pred, average="macro") # 策略是各标签平均
### 使用标签样例数加权，需设置 average="weighted"

# %%
## 多类多标签分类
# 构造有正态噪声的图片
noise1 = np.random.randint(0, 100, (len(X_train), 784))
noise2 = np.random.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise1
X_test_mod = X_test + noise2
y_train_mod = X_train
y_test_mod = X_test
# 查看含噪图片
some_index = 5500
plt.axis('off')
plt.subplot(121); plt.imshow(X_test_mod[some_index].reshape(28,28),
            cmap = mpl.cm.binary),plt.axis('off')
plt.subplot(122); plt.imshow(y_test_mod[some_index].reshape(28,28),
            cmap = mpl.cm.binary),plt.axis('off')
plt.show()
# 多类多标签分类示例，分类过程等同于去噪
knn_clf.fit(X_train_mod,y_train_mod)
clean_pic = knn_clf.predict([X_test_mod[some_index]])
plt.imshow(clean_pic.reshape(28,28), cmap = mpl.cm.binary)
plt.axis('off')

# %%






