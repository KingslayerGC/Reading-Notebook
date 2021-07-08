# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 定义初始化函数
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# %%
## 卷积层
### 卷积层是CNN最重要的组成部分。卷积层中的神经元不是连接到输入图像中的每一个像素
###（神经元），而是仅仅连接到一个小矩形（称为局部感受野）内的像素。这种架构让网络
### 专注于低级特征，然后将其组装成下一隐藏层中的高级特征。
# 卷积层工作原理演示
img1 = mp.imread(r'C:\Users\Mac\Desktop\过程\课外\电子书\Hands-On Machine Learning with Scikit-Learn and TensorFlow\data\卷积层演示1.jpg')
img2 = mp.imread(r'C:\Users\Mac\Desktop\过程\课外\电子书\Hands-On Machine Learning with Scikit-Learn and TensorFlow\data\卷积层演示2.jpg')
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.imshow(img1), plt.axis('off')
plt.figure(figsize=(10, 10))
plt.subplot(212)
plt.imshow(img2), plt.axis('off')
plt.show()

# %%
## 卷积核
### 神经元的权重矩阵称为过滤器（或卷积核）。比如，过滤器是除了中间一列是1，其余是0
### 的矩阵，那么使用这个核就会忽略每一感受野内除中央线以外的一切信息。
### 之前将每个卷积层表示为一个薄的二维层，实际上它是由几个相同大小的特征映射组成的，
### 所以使用3D图表示会更加准确。不同的特征映射有不同的卷积核。下一个卷积层将使用前
### 一层所有的特征映射（即卷积核已经是3维）。简而言之，卷积层同时对其输入应用多个
### 卷积核，使其能够检测输入中的任何位置的多个特征。
# 卷积层工作原理演示
img3 = mp.imread(r'C:\Users\Mac\Desktop\过程\编程\电子书\Hands-On Machine Learning with Scikit-Learn and TensorFlow\data\卷积层演示3.jpg')
plt.figure(figsize=(8, 8))
plt.imshow(img3), plt.axis('off')
# 加载两张示例图片
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
# 构造样本集
dataset = np.array([china, flower], dtype=np.float32)
# 设置卷积参数
batch_size, height, width, channels = dataset.shape

# %%
## 卷积核处理示例
# 定义两个卷积核
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
# 中央垂直线卷积核
filters[:, 3, :, 0] = 1
# 中央水平线卷积核
filters[3, :, :, 1] = 1
# 滤波处理
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X,
                           filters, #一组卷积核
                           strides=[1,2,2,1], #步幅参数中间的两个值是垂直和水平的步幅
                           padding="SAME") #在必要时使用零填充
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})
# =============================================================================
# # 或者使用一个简单的神经网络
# reset_graph()
# X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
# conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2],
#                         padding="SAME")
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     output = sess.run(conv, feed_dict={X: dataset})
# =============================================================================
# 滤波后图像预览
for image_index in (0, 1):
    for feature_map_index in (0, 1):
        plt.figure(figsize=(7,9))
        plt.imshow(output[image_index, :, :, feature_map_index],
                   cmap="gray", interpolation="nearest")
        plt.axis('off')

### 在使用零填充时，将在输入周围尽可能均匀地添加零。输出神经元的数量等于输入神经元
### 的数量除以步幅，向上舍入。

# %%
## 池化层
### 池化层的目标是对输入图像进行二次抽样（即收缩）以减少计算负担，内存使用量和参数
### 数量（从而限制过拟合）。就像在卷积层中一样，池化层中的每个神经元都连接到前一层
### 一个小的矩形感受野内的神经元。但不同的是，这次汇集的神经元没有权重，而是使用聚
### 合函数（如最大值或平均值）来聚合输入。
# 一个最大池化层工作原理演示
img4 = mp.imread(r'C:\Users\Mac\Desktop\过程\编程\电子书\Hands-On Machine Learning with Scikit-Learn and TensorFlow\data\池化层演示.jpg')
plt.figure(figsize=(8, 8))
plt.imshow(img4), plt.axis('off')
# 最大池化层处理示例
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], #池的四维
                          strides=[1,2,2,1], #中间两个为步幅
                          padding="VALID") #不使用零填充
with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})
plt.imshow(output[0].astype(np.uint8))
plt.show()

### 使用平均池只需将max_pool换为avg_pool。

# %%
## CNN
### 典型的 CNN 体系结构：开始于一些卷积层（每一个通常跟着一个 ReLU 层），然后是一个
### 池化层，然后又是卷积层（+ ReLU），然后池化层，等等。图像随网络变得越来越小，但是
### 由于卷积层的缘故，图像通常也会越来越深（即更多的特征映射）。在堆栈的顶部，添加由
### 全连接层（+ ReLU）组成的常规前馈神经网络，最后由最终层输出预测。
# CNN结构演示
img5 = mp.imread(r'C:\Users\Mac\Desktop\过程\编程\电子书\Hands-On Machine Learning with Scikit-Learn and TensorFlow\data\CNN架构演示.jpg')
plt.figure(figsize=(10, 10))
plt.imshow(img5), plt.axis('off')

# %%
## LeNet-5
### LeNet-5是最广为人知的 CNN 架构，主要用于手写数字识别。
### 1.MNIST 图像被零填充到 32×32 像素，并且在被输入到网络之前被归一化。其余部分不使
### 用任何填充。
### 2.平均池化层比通常的要复杂一些：每个神经元计算输入的平均值，然后将结果乘以一个可
### 学习的系数并添加一个可学习的偏差项（均为每个特征映射一个），最后应用激活函数。
### 3.C3 图中的大多数神经元仅在三个或四个 S2 图（而不是全部六个）中连接到神经元。
### 4.输出层的每个神经元不是计算输入和权向量的点积，而是输出其输入向量和其权向量之
### 间的欧氏距离平方。每个输出表达了图像属于特定数字类别的可能性大小。交叉熵损失函
### 数现在是首选，因为它更多地惩罚不好的预测，产生更大的梯度从而加快收敛。
# LeNet-5结构表
img6 = mp.imread(r'C:\Users\Mac\Desktop\过程\编程\电子书\Hands-On Machine Learning with Scikit-Learn and TensorFlow\data\LeNet结构.jpg')
plt.figure(figsize=(8, 8))
plt.imshow(img6), plt.axis('off')

# %%
## AlexNet
### 这个网络使用了两种正则化技术：首先将 dropout（50%）应用于 F8和 F9的输出。其次，
### 对训练图像进行数据增强。AlexNet还在层 C1和 C3的 ReLU 步骤之后使用局部响应标准
### 化（local response normalization）。
img7 = mp.imread(r'C:\Users\Mac\Desktop\过程\编程\电子书\Hands-On Machine Learning with Scikit-Learn and TensorFlow\data\AlexNet结构.jpg')
# AlexNet结构表
plt.figure(figsize=(8, 8))
plt.imshow(img7), plt.axis('off')

# %%
## 其他
### 其他成功的CNN还有GoogLeNet、ResNet等
# MINST完整代码（未读）
height = 28
width = 28
channels = 1
n_inputs = height * width
conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"
conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"
pool3_fmaps = conv2_fmaps
n_fc1 = 64
n_outputs = 10
reset_graph()
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
n_epochs = 10
batch_size = 100
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)

### TensorFlow 还提供了一些其他类型的卷积层：
### 1.conv1d()：为 1D 输入创建一个卷积层。
### 2.conv3d()：创建一个 3D 输入的卷积层，如 3D PET 扫描。
### 3.atrous_conv2d()：创建一个 atrous 卷积层。
### 4.conv2d_transpose()：创建一个转置卷积层，有时称为去卷积层。它通过在输入之间插入
### 零来实现卷积的逆运算，所以可以把它看作是一个使用分数步长的普通卷积层。
### 5.depthwise_conv2d()：创建一个深度卷积层，将每个卷积核独立应用于每个单独输入通道。
### 6.separable_conv2d()：创建一个可分离的卷积层。

# %%


























