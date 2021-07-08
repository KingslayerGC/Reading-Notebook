# %%
## 读取数据
import os
import pandas as pd
os.chdir(r"C:/Users/Mac/Desktop")
oridata = pd.read_csv("air_data.csv")

## 筛选出老客户
data = oridata.loc[oridata['FLIGHT_COUNT']>6]

## 定义各类型客户
data['kind'] = data['L1Y_Flight_Count'] / data['P1Y_Flight_Count']
ind1 = data['kind']>=0.9
ind2 = (data['kind']>=0.5)&(data['kind']<0.9)
ind3 = data['kind']<0.5
data['kind'].loc[ind1] = "A"
data['kind'].loc[ind2] = "B"
data['kind'].loc[ind3] = "C"

## 仅保留需要的数据
data.rename(columns={'FFP_TIER':'tier', 'AVG_INTERVAL':'avggap',
                     'avg_discount':'discount', 'EXCHANGE_COUNT':'exchange',
                     'Point_NotFlight':'point'}, inplace=True)
data['avgprice'] = (data['SUM_YR_1']+data['SUM_YR_1'])/data['SEG_KM_SUM']
data['avgpoints'] = data['Points_Sum']/data['SEG_KM_SUM']
data = data[['tier','avggap','discount','exchange','point','avgprice','avgpoints','kind']]
data.dropna(inplace=True)

## 分层划分为训练集和测试集
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(test_size=0.2)
for ind1,ind2 in split.split(data, data['kind']):
    train = data.iloc[ind1]
    test = data.iloc[ind2]

## 将训练集和测试集分别导入MYSQL
from sqlalchemy import create_engine
connect = create_engine('mysql+pymysql://root:6188@localhost:3306/MYSQL?charset=utf8')
train.to_sql('train',connect,if_exists='replace',index=False,chunksize=100)
test.to_sql('test',connect,if_exists='replace',index=False,chunksize=100)

# %%
## 从MYSQL读取数据
connect = create_engine('mysql+pymysql://root:6188@localhost:3306/MYSQL?charset=utf8')
trainset = pd.read_sql('train', connect)
testset = pd.read_sql('test', connect)
X_train = trainset.iloc[:, :7]
y_train = trainset['kind']
X_test = testset.iloc[:, :7]
y_test = testset['kind']

## 数据标准化
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)
encode = OrdinalEncoder(dtype=int)
encode.fit(y_train.values.reshape(-1,1))
y_train = encode.transform(y_train.values.reshape(-1,1))
y_test = encode.transform(y_test.values.reshape(-1,1))

# %%
## 网格搜索SVM的最佳参数，并得到对应的cv误判率
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
# 设置cv折数，定义参数表格
fold = KFold(n_splits=10, random_state=42)
param_range = {'C': [0.1, 1, 10, 100, 1000],
               'gamma': np.logspace(-1, 4, 6)}
# 进行网格搜索
svm_clf = GridSearchCV(SVC(kernel='rbf', max_iter=200), cv=fold,
                       param_grid=param_range)
svm_clf.fit(X_train, y_train)
# 得到所有组合的cv误判和最佳参数组合
svm_mcr = pd.DataFrame(
    (svm_clf.cv_results_['mean_test_score']).reshape(5, -1),
    columns=param_range['gamma'], index=param_range['C'])
print("高斯核SVM各参数组合的cv准确率表，行为C值，列为gamma值\n", svm_mcr)

# %%
## 决策树分类器
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=20)
tree_clf.fit(X_train, y_train)
# 可视化
import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(
    tree_clf, out_file=None, max_depth=2, rotate=True,
    feature_names=['tier','avggap','discount',
                   'exchange','point','avgprice','avgpoints'],
    filled=True, rounded=True,  special_characters=True)  
graphviz.Source(dot_data)

# %%
## BP神经网络
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
tf.reset_default_graph()
# 开始架设神经网络
n_inputs = 1*7
n_hidden1 = 150
n_hidden2 = 150
n_outputs = 3
learning_rate = 0.1
n_epochs = 40
batch_size = 100
# 使用占位符节点来表示训练数据和目标
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
# 创建隐藏层和输出层
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.elu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.elu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)
# 使用交叉熵计算代价函数
with tf.name_scope("loss"):
    # 根据通过softmax之前的输出和整数形式的标签计算交叉熵，输出一个交叉熵张量。
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    # 计算交叉熵张量的平均数作为代价函数
    loss = tf.reduce_mean(xentropy, name="loss")
# 定义训练模式
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
# 评估神经网络
with tf.name_scope("eval"):
    # 将每个样本预测概率前k的标签与真实标签作对比，输出一个布尔张量
    correct = tf.nn.in_top_k(logits, y, k=1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# 创建初始化变量节点
init = tf.global_variables_initializer()
# 创建保存节点
saver = tf.train.Saver()
# 开始训练神经网络
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        # 随机梯度下降一次迭代
        for X_batch, y_batch in shuffle_batch(X_train, y_train.ravel(), batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        # 每次迭代进行一次评估
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test.ravel()})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)
    save_path = saver.save(sess, r'./Tensorflow Model\guo_model.ckpt')
# 使用神经网络进行预测
with tf.Session() as sess:
    saver.restore(sess, r'./Tensorflow Model\guo_model.ckpt')
    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

# %%
## 计算混淆矩阵
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def confuse_mat(y_true, y_pred):
    confuse_mat = confusion_matrix(y_true, y_pred)
    # 比较错误率而不是错误数
    row_sums = confuse_mat.sum(axis=1, keepdims=True)
    norm_confuse_mat = confuse_mat / row_sums
    # 排除对角线的干扰
    np.fill_diagonal(norm_confuse_mat, 0)
    return norm_confuse_mat
## 绘制混淆图
fig, axes = plt.subplots(1, 3, figsize=(9,9))
axes[0].matshow(confuse_mat(y_train, svm_clf.predict(X_train)), cmap=plt.cm.gray)
axes[0].set_title("SVM", y=-0.2)
axes[1].matshow(confuse_mat(y_train, tree_clf.predict(X_train)), cmap=plt.cm.gray)
axes[1].set_title("Decision Tree", y=-0.2)
with tf.Session() as sess:
    saver.restore(sess, r'./Tensorflow Model\guo_model.ckpt')
    Z = logits.eval(feed_dict={X: X_train})
    y_pred = np.argmax(Z, axis=1)
axes[2].matshow(confuse_mat(y_train, y_pred), cmap=plt.cm.gray)
axes[2].set_title("Neural Network", y=-0.2)

# %%
## 准确率示意图
def score(clf):
    return [clf.score(X_train, y_train), clf.score(X_test, y_test)]
with tf.Session() as sess:
    saver.restore(sess, r'./Tensorflow Model\guo_model.ckpt')
    s1 = accuracy.eval(feed_dict={X: X_train, y: y_train.ravel()})
    s2 = accuracy.eval(feed_dict={X: X_test, y: y_test.ravel()})
accuscore = pd.DataFrame([score(svm_clf), score(tree_clf), [s1,s2]],
                        columns=['Train Set', 'Test Set'],
                        index=['SVM', 'Decision Tree', 'BP Network']
                        ).stack().reset_index()
accuscore.columns=['Model', 'Set', 'Accuracy']
import seaborn as sns
sns.set()
sns.barplot(x='Model', y='Accuracy', hue='Set', data=accuscore)
