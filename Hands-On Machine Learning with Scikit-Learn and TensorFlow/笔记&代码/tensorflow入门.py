# %%
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.__version__

# %%
### TensorFlow 程序通常分为两部分。
### 第一部分构建计算图谱（构造阶段），第二部分运行它（执行阶段）。
## 创建计算图谱
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
# 创建多个计算图谱
graph = tf.Graph()
# 将graph展示设置为默认图谱来加入x2变量
with graph.as_default():
    x2 =  tf.Variable(2)
# x2所属图谱不再为默认图谱
x2.graph is tf.get_default_graph()
# 重置默认图谱
tf.reset_default_graph() 

# %%
## 创建会话来计算
# 方法 1,创建一个会话
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
sess.close() #需手动关闭
# 方法 2，设置会话为默认会话，将自动关闭
with tf.Session() as sess:
    x.initializer.run() #等效于调用 tf.get_default_session().run(x.initializer)
    y.initializer.run()
    result = f.eval() #等效于调用 tf.get_default_session().run(f) 
# 方法 3，使用变量一步初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
# 方法 4，使用InteractiveSession()，它将把自身设置为默认对话
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
result = f.eval()
sess.close() #需手动关闭

# %%
## 节点的生命周期
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
    y.eval() #一个图形运行后，所有节点值将删除
    z.eval() #需要重新计算w和x
# 减少重复计算
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])

### 在单进程 TensorFlow 中，多个会话不共享任何变量，即使它们复用同一个图。 
### 在分布式 TensorFlow 中，变量状态存储在服务器上，因此多个会话可以共享相同的变量。

# %%
## 线性回归tensorflow示例:加州房价
# 获取数据
import numpy as np
import pandas as pd
data = pd.read_csv(r"C:\Users\Mac\Desktop\过程\项目\电子书项目\Sklearn 与 TensorFlow 机器学习实用指南\房价\housing.csv",sep=',',header=0).dropna(subset=["total_bedrooms"])
housing = data.drop(['median_house_value', 'ocean_proximity'], axis=1).values
housing_value = np.array(data['median_house_value'])
m,n = housing.shape
housing_plus_bias = np.c_[np.ones((m, 1)), housing]
# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housing_scaled = scaler.fit_transform(housing)
housing_scaled_plus_bias = np.c_[np.ones((m, 1)), housing_scaled]

# %%
## 正规方程法
# 定义计算图谱
X = tf.constant(housing_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing_value.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT),y)
# 打开会话计算
with tf.Session() as sess:
    theta_value = theta.eval()

# %%
## 梯度下降法
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(housing_scaled_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing_value.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0),
                    name="theta") #生成包含随机值的张量
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
# 计算张量各个维度上的元素的平均值
mse = tf.reduce_mean(tf.square(error), name="mse")
# 手动计算梯度
gradients = 2/m * tf.matmul(tf.transpose(X), error)
# 创建一个为变量分配新值的节点,即迭代过程
training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
        best_theta = theta.eval()
# 自动计算梯度（替换 gradients = ... 行）
gradients = tf.gradients(mse, [theta])[0]
# 使用梯度下降优化器（替换 gradients = ... 和 training_op = ... 行）
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
# 使用动量优化器（通常比渐变收敛快得多）
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

# %%
## 小批量梯度下降法
# 占位符节点示例
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
# 小批量梯度下降
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# 仅保存 theta 变量，键名称是 weights
#saver = tf.train.Saver({"weights": theta})
n_epochs = 10
batch_size = 100 #一批次样本数
n_batches = int(np.ceil(m / batch_size))
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = housing_scaled_plus_bias[indices]
    y_batch = housing_value.reshape(-1, 1)[indices]
    return X_batch, y_batch
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()
    # 保存模型
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")

# %%
## 加载保存的模型
# 重置默认图谱
tf.reset_default_graph()
# 加载保存的计算图谱（可以代替以上构建计算图谱的代码）
saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")
theta = tf.get_default_graph().get_tensor_by_name("theta:0")
# 调用保存的模型
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval()
# 结果与原模型一致
np.allclose(best_theta, best_theta_restored)

# %%
## 创建名称作用域来对节点进行分组
# 定义名为 loss 的名称作用域内的节点 error和 mse
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
# 节点名称将以 “loss/” 为前缀
mse.op.name

# %%
## tensorboard
# 加!的为改动行
from datetime import datetime #!
now = datetime.utcnow().strftime("%Y%m%d%H%M%S") #!
root_logdir = "tf_logs" #!
logdir = "{}/run-{}/".format(root_logdir, now) #!
n_epochs = 1000
learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
with tf.name_scope("loss") as scope: #!
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE', mse) #!
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph()) #!
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = housing_scaled_plus_bias[indices]
    y_batch = housing_value.reshape(-1, 1)[indices]
    return X_batch, y_batch
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0: #!
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch}) #!
                step = epoch * n_batches + batch_index #!
                file_writer.add_summary(summary_str, step) #!
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        best_theta = theta.eval()       
file_writer.flush() #!
file_writer.close()

### 创建日志完成后，按以下步骤打开Tensorboard
### 打开cmd，输入tensorboard --host=127.0.0.1 --logdir=C:\Users\Mac\tf_logs\run-20200212100740
### 注意地址无需用字符串写法，然后在浏览器打开显示的网址即可

# %%
## 模块性
# 创建五个 ReLU（整流线性单元）并输出其总和
def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, 0., name="relu")
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

### 创建节点时，TensorFlow 将检查其名称是否已存在，如果已经存在，
### 则会附加一个下划线，后跟一个索引，以使该名称是唯一的。
### TensorBoard 识别这样的系列并将它们折叠在一起以减少混乱。

# %%
## 在图形的各个组件之间共享变量
#  控制 ReLU 的阈值（以下方法一、方法二代码仅为修改位置）
# 方法一：创建一个变量节点来控制阈值
def relu(X, threshold):
    return tf.maximum(z, threshold, name="max")
threshold = tf.Variable(0.0, name="threshold")
relus = [relu(X, threshold) for i in range(5)]
# 方法二：将阈值变量设置为函数的属性
with tf.name_scope("relu"):
    if not hasattr(relu, "threshold"):
        relu.threshold = tf.Variable(0.0, name="threshold")
relus = [relu(X, relu.threshold) for i in range(5)]
# 方法三：创建共享变量,使用函数前初始化该变量
tf.reset_default_graph()
def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="max")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    # 阈值初始化
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
    relus = [relu(X) for relu_index in range(5)]
    output = tf.add_n(relus, name="output")
# 方法四：创建并初始化共享变量，仅在第一次调用时设置 reuse = False 
tf.reset_default_graph()
def relu(X):
    # 定义阈值时已经初始化
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, threshold, name="max")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")

# %%
## 










