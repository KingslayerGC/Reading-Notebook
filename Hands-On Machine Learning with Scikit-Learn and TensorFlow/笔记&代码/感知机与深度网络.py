# %%
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# %%
## 感知器
### 感知器是最简单的人工神经网络结构之一，基于线性阈值单元（LTU），使用阶跃函数。
### 感知器一次对一个训练实例进行预测，然后对产生错误预测的输出神经元加强输入的连接权重。
### 感知器学习算法类似于随机梯度下降。
### 缺陷：有所有线性分类器的缺陷
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int)
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)
# 感知机不输出类概率，而是基于硬阈值进行预测
y_pred = per_clf.predict([[2, 0.5]])
# 下面的随机梯度下降与以上感知机基本相同
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier( loss="perceptron", learning_rate="constant",
                        eta0=1, penalty=None)
sgd_clf.fit(X, y)

# %%
## 多层感知器
### 多层感知器(MLP)由一个输入层、若干个隐藏层LTU和一个输出层LTU组成。
###　当神经网络有两个或多个隐含层时，称为深度神经网络(DNN)。
## 反向传播训练
### 对于每个样本，反向传播算法首先进行预测并计算误差（前向），然后反向遍历每个层
### 来测量每个连接的误差贡献，最后调整连接器权值（梯度下降步长）以减少误差。简单的
### MLP通常用Logistic函数代替阶跃函数，因为 Logistic 函数到处都有一个定义良好的非零
### 导数，允许梯度下降在每一步上都取得进展。
### 另外两个流行的激活函数是双曲正切函数和Relu函数。
### 双曲正切函数：就像 Logistic 函数，它是 S 形的、连续的、可微的，但是它的输出值
### 范围是从-1到 1，这使每个层的输出在训练开始时基本都正则化了（以 0 为中心）。这有
### 助于加快收敛速度。
### Relu 函数：它是连续的，但在 z=0 时不可微（斜率突然改变可能使梯度下降反弹）。
### 然而在实践中它工作得很好，并且具有快速计算的优点。最重要的是，它没有最大输出值
### 的事实也有助于减少梯度下降期间的一些问题。
### MLP 通常用于分类，每个输出对应于不同的二进制类。当类是多类的时，输出层通常使用
### 共享的softmax函数替换单独的激活函数来使每个神经元的输出对应于相应类的估计概率。
# 生成mnist数据
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# %%
## 使用DNNClassifier训练有两个隐藏层和一个SOFTMax输出层的DNN分类器
feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], #隐藏层神经元数
                                     n_classes=10, #输出层神经元数
                                     activation_fn=None, # 默认使用Relu
                                     feature_columns=feature_cols)
# 设置一个object参数
input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_train}, y=y_train,
        num_epochs=40, batch_size=50, shuffle=True) #随机梯度下降
dnn_clf.train(input_fn=input_fn)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
# 生成评估结果
dnn_clf.evaluate(input_fn=test_input_fn)
# 生成详细预测结果（迭代器形式）
y_pred=list(dnn_clf.predict(input_fn=test_input_fn))

# %%
## 自定义相同的模型
tf.reset_default_graph()
# 开始架设神经网络
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01
n_epochs = 40
batch_size = 50
# 使用占位符节点来表示训练数据和目标
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
# 图层创建函数，需要参数来指定输入神经元数量，激活函数和图层的名称
def neuron_layer(X, n_neurons, name, activation=None):
    # 使用图层名称来创建名称作用域
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        # 创建一个截断的正态分布（不产生与均值距离超过两倍标准差的值）
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        # 创建一个保存权重矩阵的 W 变量
        W = tf.Variable(init, name="kernel")
        # 创建偏置神经元
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
# 创建隐藏层和输出层（未经过softmax计算处理）
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")
# 或使用 API dense 来创建图层，效果与上面一致
'''from tensorflow.layers import dense
with tf.name_scope("dnn"):
    hidden1 = dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = dense(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)'''
# 使用交叉熵计算代价函数
with tf.name_scope("loss"):
    # 根据通过softmax之前的输出和整数形式的标签计算交叉熵，输出一个交叉熵张量。
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y,logits=logits)
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
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        # 每次迭代进行一次评估
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)
    save_path = saver.save(sess, r'./Tensorflow Model\my_DNN_model.ckpt')
# 使用神经网络进行预测
with tf.Session() as sess:
    saver.restore(sess, r'./Tensorflow Model\my_DNN_model.ckpt')
    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
# 对比预测标签和真实标签
print("Predicted classes:", y_pred)
print("Actual classes:   ", y_test[:20])

### 所有隐藏层都需要随机初始化连接权重，以避免因对称性出现梯度下降算法无法终止的情况。
### 使用截断的正态分布而不是常规正态分布确保不会有任何大的权重(大权值会减慢训练)。
### 这是对神经网络的微小调整之一，对它的效率产生了巨大的影响。
### sparse_softmax_cross_entropy_with_logits()函数等同于先应用 SOFTMAX 激活函数，
### 然后计算交叉熵，但它比SOFTMAX更高效。
### 还有一个函数softmax_cross_entropy_with_logits()，用于标签单热形式的标签。

# %%
## 寻找最佳的参数
### 隐藏层数：深层网络可以使用比浅网络更少的神经元来建模复杂的函数，使得训练更快。
### 隐藏层的神经元数量：可以逐渐增加神经元的数量，直到网络开始过度拟合。
### 一个方法是选择一个具有比实际需要更多层次和神经元的模型，然后使用早停
### （以及其他正则化技术）来防止它过度拟合。
### 激活函数：在大多数情况下，在隐藏层中使用 ReLU 激活函数（或其中一个变体）。
### 与其他激活函数相比，Relu 计算速度要快一些，而且在logits的高原地区不会被卡住，
### 因为它不会对大的输入值饱和（相对地，逻辑函数或双曲正切函数容易在1附近饱和)
### 对于输出层，softmax 激活函数通常是分类任务的良好选择。
### 对于回归任务则可以不使用激活函数。



































