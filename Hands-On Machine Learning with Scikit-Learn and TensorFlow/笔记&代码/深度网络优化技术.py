# %%
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 获取数据
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
# 定义选取一批次数据的函数
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
# 定义初始化函数
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# %%
## 梯度爆炸/梯度消失
### 梯度不稳定是深度神经网络的一大问题，不同的层次可能以相距甚大的速度学习。
### 梯度消失：随着反向传播算法进行到较低层，梯度变得越来越小。这导致低层连接权重
### 实际上保持不变，参数可能永远不会收敛到良好的状态。
### 梯度爆炸：梯度变得越来越大，许多层得到了非常大的权重更新，算法呈发散姿态。这个
### 问题在循环神经网络中最为常见。

# %%
## 随机初始化
### Xavier Glorot发现了sigmoid激活函数和标准正态分布的权重初始化的组合有重大缺陷。
### 梯度爆炸：网络正向传播时，直至激活函数在顶层达到饱和，每层的方差均持续增加，而
### logistic函数的平均值是0.5而不是0使得情况变得更糟。因此双曲正切函数的表现就要略好
### 于logistic函数。
### 梯度消失：函数在 0或 1饱和后，导数非常接近 0。因此当反向传播开始时，小梯度不断地
### 被稀释，几乎没有梯度能通过网络回传，较低层权重也不会发生变化。同样地，双曲正切
### 函数由于具有更大的导数范围，表现要略好。
### 在继续采用sigmoid激活函数的情况下，可以采用Xavier初始化来初始化连接权重。
### 后来针对不同的激活函数，有了类似Xavier的策略，比如ReLU激活函数采用He初始化。
### dense()默认使用 Xavier初始化
# =============================================================================
# # 改为使用He初始化
# he_init = tf.variance_scaling_initializer()
# hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
#                           kernel_initializer=he_init, name="hidden1")
# =============================================================================

# %%
## 非饱和激活函数
### ReLU 激活函数在深度神经网络中表现得好，主要是因为它对正值不会饱和，有助于减轻
### 梯度爆炸；导数为常数，有助于减轻梯度消失的问题；并且它的计算速度很快。但它并不
### 完美，它有一个被称为 “ReLU 死区” 的问题：在训练过程中，如果神经元输入为负，则梯
### 度为 0，它将一直输出 0，这意味着神经元的死亡。如果使用大的学习率，情况将更加严重。 
### 解决方法：使用 ReLU 函数的变体
### Leaky ReLU：这个函数定义为 max(αz，z)，超参数 α 是 z < 0 时函数的斜率，通常设置
### 为 0.01。这个小斜坡确保 leaky ReLU 永不死亡（仍然可能长期昏迷）。一般来说，
### Leaky Relu 总是优于普通 ReLU 激活函数。事实上，设定更大的leak（如 α=0.2）似乎
### 比小 leak有更好的性能。
### 随机化 Leaky ReLU(RReLU)：其中 α 在训练期间在给定范围内随机挑选，并在测试期间
### 固定为平均值，它表现很好，且是一个正则项，可以减少训练集的过拟合风险。
### 参数 Leaky ReLU(PReLU)：将 α 改为可以被反向传播修改的参数，在训练期间被学习，
### 它在大型图像数据集上的表现强于普通ReLU，但是对于较小的数据集有过度拟合的风险。
### 指数线性单元（exponential linear unit，ELU）：超参数 α 定义为函数的负无穷极限。
### α 通常设置为 1，可以随时调整。它的平均输出接近于 0，有助于减轻梯度消失问题；计算
### 速度慢于 ReLU 及其变体（因此在测试时较慢），但有更快的收敛速度，能减少训练时间，
### 且在测试集上表现更好；在任何地方都是平滑的，有助于加速梯度下降。 
### 总结：一般 ELU > leaky ReLU 及其变体 > ReLU > tanh > sigmoid。 如果您关心
### 系统运行时的性能，那么可以选择 leaky ReLU。如果担心神经网络过拟合，可以选
### 择 RReLU; 如果拥有庞大的训练数据集，则为 PReLU。
# =============================================================================
# # 修改激活函数为 Leaky ReLU（需要自定义）
# def leaky_relu(z, name=None):
#     return tf.maximum(0.01 * z, z, name=name)
# hidden1 = tf.layers.dense(X, n_hidden1,
#                           activation=leaky_relu, name="hidden1")
# # 修改激活函数为 ELU
# hidden1 = tf.layers.dense(X, n_hidden1,
#                               activation=tf.nn.elu, name="hidden1")
# =============================================================================
### 本书没有出现 SELU 。它在ELU的基础上乘以一个系数lambda。经过该激活函数后，输出
### 将自动归一化到 0 均值和 1 方差，保证梯度不会爆炸或消失，效果比以上任何激活函数
### 乃至接下来的批量标准化都要更好。它的唯一缺点是自规范属性容易被破坏。

# %%
## 批量标准化
### 批量标准化可以解决每层输入的分布在训练期间改变的问题。
### 该技术在每层的激活函数之前对输入进行标准化，然后每层使用两个新参数分别对对结果进行
### 尺度变换和偏移，这个操作称为反标准化，可以让模型学习到每层输入值的最佳均值和尺度。
### 在训练中，使用的均值和标准差是当前小批量输入的均值和标准差；在测试时，则使用
### 整个训练集的均值和标准差。
### 批量标准化会增加模型的复杂性，并降低预测速度。然而，由于梯度不稳定问题的好转，
### 模型现在可以使用大学习率，从而加快训练速度。另外，BN也是一个弱正则项。
# 使用partial能一次性设置相同的参数
from functools import partial
reset_graph()
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01
n_epochs = 20
batch_size = 200
# 开始架设神经网络
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')
# 设置
with tf.name_scope('dnn'):
    # 设置批量标准化
    training = tf.placeholder_with_default(False, shape=(), name='training')
    my_batch_norm_layer = partial(tf.layers.batch_normalization,
                                  training=training, momentum=0.9)
    # 设置图层创建，使用He初始化
    he_init=tf.variance_scaling_initializer()
    my_dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)
    # 创建图层
    hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1, n_hidden2, name='hidden2')
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
    logits_before_bn = my_dense_layer(bn2, n_outputs, name='outputs')
    logits = my_batch_norm_layer(logits_before_bn)
    y_proba = tf.nn.softmax(logits)
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# 训练神经网络
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            # 运行批量标准化需要额外的更新操作
            sess.run([training_op, extra_update_ops],
                     feed_dict={training:True, X:X_batch, y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={X:X_valid, y:y_valid})
        print(epoch, 'Validation accuracy:', accuracy_val)
    save_path=saver.save(sess, r'./Tensorflow Model\my_batchnorm_model.ckpt')
# =============================================================================
# # 或者换成以下方法，然后只需要在训练过程中运行training_op，TensorFlow会自动运行更新操作
# with tf.name_scope("train"):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(extra_update_ops):
#         training_op = optimizer.minimize(loss)
# =============================================================================

# %%
## 梯度裁剪
# 设置阈值为 1（和-1）
threshold = 1.0
# 调用计算出的梯度
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
# 进行裁剪
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
              for grad, var in grads_and_vars]
# 应用裁剪完的梯度
training_op = optimizer.apply_gradients(capped_gvs)

# %%
## 复用预训练层
### 从零开始训练一个大的 DNN 通常不是一个好主意，相反应该尝试找到一个能够完成类似任务
### 的现有的神经网络，复用这个网络的较低层。这称作迁移学习，会大大加快训练速度，并且
### 不需要很多的训练数据。任务越相似，可以重复使用的层就越多。
# 加载保存的图层来确认你需要重建哪些变量节点
reset_graph()
saver = tf.train.import_meta_graph(
        r'./Tensorflow Model\my_batchnorm_model.ckpt.meta')
# 打印所有节点
for op in tf.get_default_graph().get_operations():
    print(op.name)
# 重建所需要的节点
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
training_op = tf.get_default_graph().get_operation_by_name("train/GradientDescent")
# =============================================================================
# # 或者在构建原始模型时做如下操作
# for op in (X, y, accuracy, training_op):
#     tf.add_to_collection("my_important_ops", op)
# # 简单地重建所需要的节点
# X, y, accuracy, training_op = tf.get_collection("my_important_ops")
# =============================================================================
# 复用模型，相当于继续训练
with tf.Session() as sess:
    # 复用操作
    saver.restore(sess,  r'./Tensorflow Model\my_batchnorm_model.ckpt')
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
    #save_path = saver.save(sess, "./my_new_model_final.ckpt") 
# =============================================================================
# # 仅复用两个隐藏层（部分复用）
# reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
#                                scope="hidden[12]")
# # 创建一个恢复字典，key是原模型节点名，值是现模型节点名
# reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
# # 创建一个Saver，它将只恢复这些变量
# restore_saver = tf.train.Saver(reuse_vars_dict)
# # 局部复用仍需初始化
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     # 复用操作
#     restore_saver.restore(sess,  r'./Tensorflow Model\my_batchnorm_model.ckpt')
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#         print(epoch, "Validation accuracy:", accuracy_val)
#     #save_path = saver.save(sess, "./my_new_model_final.ckpt")
# =============================================================================

# %%
## 复用其他框架的模型
# 找到需要喂给的节点
graph = tf.get_default_graph()
assign_kernel = graph.get_operation_by_name("hidden1/kernel/Assign")
assign_bias = graph.get_operation_by_name("hidden1/bias/Assign")
init_kernel = assign_kernel.inputs[1]
init_bias = assign_bias.inputs[1]
# 将外部值喂给节点
with tf.Session() as sess:
    sess.run(init, feed_dict={init_kernel: original_w, init_bias: original_b})

# %%
## 冻结层
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # 只训练第二层
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope="hidden[2]|outputs")
    training_op = optimizer.minimize(loss, var_list=train_vars)
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[1]")
restore_saver = tf.train.Saver(reuse_vars)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    restore_saver(sess, r'./Tensorflow Model\my_batchnorm_model.ckpt')
    # 先计算冻结层输出
    h1_cache = sess.run(hidden1, feed_dict={X:X_train})
    h1_cache_valid = sess.run(hidden1, feed_dict={X:X_valid})
    # 直接将冻结层输出用于训练
    for epoch in range(n_epochs):
        shuffled_index = np.random.permutation(len(X_train))
        h1_batches = np.array_split(h1_cache[shuffled_index], n_batches)
        y_batches = np.array_split(y_train[shuffled_index], n_batches)
        for h1_batch, y_batch in zip(h1_batches, y_batches):
            sess.run(training_op, feed_dict={hidden1:h1_batch, y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={X:X_valid, y:y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)
        #save_path = saver.save(sess, "./my_new_model_final.ckpt")

### 先确定合适的复用层数。然后尝试冻结所有复制层，训练模型并查看效果。接着尝试解冻
### 一两个较高隐藏层，让反向传播调整它们，检查性能是否提高。拥有的训练数据越多，可
### 以解冻的层数就越多。
### 关于模型来源：TensorFlow 在 https://github.com/tensorflow/models 中有自己的
### Model Zoo。它包含了大多数最先进的图像分类网络，包括代码、预训练模型和流行的图
### 像数据集。还有 Caffe Model Zoo。它包含许多在各种数据集上训练的计算机视觉模型，
### 详情查看https://github.com/ethereon/caffetensorflow。

# %%
## 其他预训练方法
### 假设没有很多的有标记训练数据，且不能找到一个类似的任务训练模型。
### 可以进行无监督的训练。可以尝试在冻结其他所有层的情况下逐层训练，从最低层开始一直
### 向上，使用无监督的特征检测算法，如限制玻尔兹曼机(RBM)或自动编码器。每个层的训练
### 数据都来自先前训练过的层的输出。所有层进行完训练后，就可以使用监督式学习（即反
### 向传播）对网络进行微调。这是一个相当漫长而乏味的过程，但通常运作良好。
### 另一个选择是提出一个监督的任务，例如，如果要训练一个模型来识别图片中的朋友，而又
### 没有太多朋友的照片，此时可以先在互联网上下载数百万张脸部照片并训练一个分类器来检
### 测两张脸是否相同，然后使用直接使用此分类器或复用它的较低层。
    
# %%
## 优化器
### 训练一个非常大的深度神经网络可能会非常缓慢。到目前为止，我们已经看到了四种加速训
### 练的方法（并且达到更好的效果）：对连接权重应用良好的初始化策略，使用良好的激活
### 函数，使用批量规范化以及重用预训练网络的低级部分。
### 接下来是另一种巨大的速度提升方法，使用比普通渐变下降优化器更快的优化器。
# 动量优化器，模拟真实物体滑落
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9) #摩擦系数
# Nesterov 加速梯度，动量优化的变体，几乎总是比普通动量优化器效果好
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9, use_nesterov=True)
# AdaGrad
### AdaGrad具有自适应学习率，但不幸的是，在训练神经网络时由于学习率被缩减得太多，它
### 经常停在达到全局最优之前。所以，不应该用它来训练深度神经网络。
# RMSProp，AdaGrad的改进版本，通常比动量优化更好，一度是首选的优化算法
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                      momentum=0.9,
                                      decay=0.9, # 衰变率
                                      epsilon=1e-10)
# Adam，自适应矩估计，结合了动量优化和 RMSProp的思想
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

### 以上都有未列出的超参数，一般使用默认值即可。
### 结论：几乎总是应该使用 Adam_optimization，训练通常会快好几倍。
### 除了这些一阶偏导算法，还有二阶偏导算法，但不适用于深度神经网络。

# %%
## 稀疏模型
### 如果需要一个非常快速的模型，或者需要它占用较少的内存，可以使用一个稀疏模型。
### 实现的一种方法是：直接摆脱微小的权重（将它们设置为 0）。
### 另一个方法是：在训练过程中应用强 l1 正则化。
### 可以使用遵循正则化领导（FTRL）的技术，当与 l1正则化一起使用时，这种技术通常导致
### 非常稀疏的模型。TensorFlow可以实现称为 FTRL-Proximal 的 FTRL 变体。

# %%
## 学习率调整
### AdaGrad，RMSProp，Adam等自带学习率调整功能，无需再手动调整
with tf.name_scope("train"):
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    # 设置节点记录梯度下降次数
    global_step = tf.Variable(0, trainable=False, name="global_step")
    # 设置学习率衰减模式
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step, decay_steps, decay_rate)
    # 使用动量优化器
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step=global_step)

# %%
## 正则化技术
### 一些最流行的神经网络正则化技术：早期停止，l1 和 l2 正则化，drop out，最大范数
### 正则化和数据增强。
### 早期停止：在实践中运行良好，通常将其与其他正则化技术相结合，从而获得更高的性能。
# 设置l1正则化参数
scale = 0.001
# 获得权重参数
W1 = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
W2 = tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    base_loss = tf.reduce_mean(xentropy, name='avg_xentropy')
    reg_loss = tf.reduce_mean(tf.abs(W1)) + tf.reduce_mean(tf.abs(W2))
    loss = tf.add(base_loss, scale*reg_loss, name='loss')
# 或者使用如下方法
from functools import partial
# 定义带正则化选项的神经元
my_dense_layer = partial(tf.layers.dense, activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l1(scale))
with tf.name_scope('dnn'):
    hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
    hidden2 = my_dense_layer(hidden1, n_hidden2, name='hidden2')
    logits = my_dense_layer(hidden2, n_outputs, activation=None, name='outputs')
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    base_loss = tf.reduce_mean(xentropy, name='avg_xentropy')
    # 调出l1正则项
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss]+reg_loss, name='loss')

# %%
## dropout
### dropout是最流行的深度神经网络正则化技术。在每个训练步骤中，每个神经元（不包括
### 输出神经元）都有p的概率暂时被“丢弃”，即在这个训练步骤中它将被完全忽略（在下一步
### 可能会激活）。dropout会减缓训练速度，但通常会使模型效果更好。超参数 p 称为丢弃
### 率，也即1-保留率，通常设为 50%。如果模型过拟合，可以增加丢弃率；相反如果欠拟合，
### 则应降低丢弃率。
# 设置丢弃率
dropout_rate=0.5
training = tf.placeholder_with_default(False, shape=(), name='training')
# 设置正则化步骤
X_drop = tf.layers.dropout(X, dropout_rate, training=training)
with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X_drop, n_hidden1,
                              activation=tf.nn.relu, name='hidden1')
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2,
                              activation=tf.nn.relu, name='hidden2')
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name='outputs')

### 还有一种称为 Dropconnect 的 dropout 变体，方法是随机丢弃输入权重（设为 0）而不
### 是丢弃神经元输出。一般而言，dropout表现会更好。
    
# %%
## Max Norm
### 另一种流行的正则化技术是 Max Norm：对于每个神经元，它约束输入权重 w 的2范数，
### 阈值 r 称为最大范数超参数。具体做法是在每个训练步骤之后计算范数，如果超过阈值
### 则对 w 进行削减，通常是减至原值的 r/||w||2 倍。
### 最大范数正则化还可以帮助减轻梯度消失/爆炸问题（如果不使用批量标准化）。
# 根据2范数剪切
from functools import partial
threshold = 1.0
my_clip = partial(tf.clip_by_norm, clip_norm=threshold, axes=1)
# 设置剪切和替换操作节点
weights = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
clipped_weights = my_clip(weights)
clipping = tf.assign(weights, clipped_weights)
weights2 = tf.get_default_graph().get_tensor_by_name("hidden2/kernel:0")
clipped_weights2 = my_clip(weights2)
clipping2 = tf.assign(weights2, clipped_weights2)
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            # 执行剪切替换操作
            clipping.eval()
            clipping2.eval()
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", acc_valid)
    #save_path = saver.save(sess, "./my_model_final.ckpt")  
# =============================================================================
# # 或者定义一个权重正则器，返回值是一个留有收集索引的剪切替换函数
# def max_norm_regularizer(threshold, axes=1, name="max_norm",
#                          collection="max_norm"):
#     def max_norm(weights):
#         clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
#         clip_weights = tf.assign(weights, clipped, name=name)
#         tf.add_to_collection(collection, clip_weights)
#         return None #正常情况下返回正则项
#     return max_norm
# max_norm_reg = max_norm_regularizer(threshold=1.0)
# with tf.name_scope("dnn"):
#     hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
#                               kernel_regularizer=max_norm_reg, name="hidden1")
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
#                               kernel_regularizer=max_norm_reg, name="hidden2")
#     logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
# # 将剪切和替换操作节点收入合集
# clip_all_weights = tf.get_collection("max_norm")
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#             # 执行剪切替换操作
#             sess.run(clip_all_weights)
#         acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#         print(epoch, "Validation accuracy:", acc_valid)
#     #save_path = saver.save(sess, "./my_model_final.ckpt")             
# =============================================================================

# %%
## Data Augmentation
### 最后一个正则化方法是数据增广，这是一种从现有的训练实例中产生新的训练实例，人为
### 增加训练集大小的技术。这将减少过拟合。关键是要生成逼真的训练实例，例如：使用
### TensorFlow 提供的多种图像处理操作函数，对训练集中的每个图片进行移动，旋转或调整
### 大小等操作，并将新图片添加到训练集。这些新的训练实例在训练期间生成，所以并不会浪
### 费存储空间。

# %%
## 总结
# 输出一个大部分时间都适用的策略组合
from prettytable import PrettyTable
table = PrettyTable(['Configuration','Method'])
table.add_row(['Intialization','He initialization'])
table.add_row(['Activation function','ELU'])
table.add_row(['Normalization','Batch normalization'])
table.add_row(['Regularization','Dropout'])
table.add_row(['Optimizer','Adam'])
table.add_row(['Learning rate schedule','None'])
print('Common Strategy')
print(table)

