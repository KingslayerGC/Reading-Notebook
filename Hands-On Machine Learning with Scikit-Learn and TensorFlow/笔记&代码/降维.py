# %%
import numpy as np

# %%
## 降维思想：投影 降维方法：PCA
### 主成分分析是目前为止最流行的降维算法。首先它找到接近数据集分布的超平面，然后将所
### 有的数据都投影到这个超平面上。选择超平面的依据是，使得将原始数据集投影到该超平面
### 上的均方距离（方差）最小。
### PCA 寻找训练集中可获得最大方差的轴。然后找到一个与第一个轴正交的第二个轴；在一个
### 更高维的数据集中，找到与前两个轴正交的第三个轴，第四个轴，第五个轴等。这些轴的单
### 位矢量称为主成分。找到主成分的方法是一种称为奇异值分解(SVD)的标准矩阵分解技术。
### PCA 假定数据集以原点为中心，sklearn的PCA类会自动对数据集做中心化处理。

# %%
## 自定义PCA
# 生成数据（没看懂）
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
# 中心化处理
X_centered= X-X.mean(axis=0)
# 奇异值分解
U,s,V=np.linalg.svd(X_centered)
# 选取前两个主成分
W2= V.T[:,:2]
# 通过计算点积将原矩阵投影到主成分所组成的超平面上
X2D1= X_centered.dot(W2)

# %%
## sklearn的PCA降维，跟自定义结果一致
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X2D2=pca.fit_transform(X)
# 访问所有主成分,每一个行为一个主成分
pca.components_
# 方差解释率
pca.explained_variance_ratio_
# 选择累计方差解释率达到某个标准的前d个主成分
pca=PCA()
pca.fit(X)
cumsum=np.cumsum(pca.explained_variance_ratio_)
d=np.argmax(cumsum>=0.95)+1
# 或者直接使用浮点数型的超参数n_components
pca=PCA(n_components=0.95)

# %%
## PCA压缩
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X_mnist= mnist["data"]
# 将原数据压缩至154维
pca= PCA(n_components=154)
X_mnist_reduced= pca.fit_transform(X_mnist)
# 解压缩，即重构数据
X_mnist_recovered= pca.inverse_transform(X_mnist_reduced)
# 对比原数据和重构数据
import matplotlib as mpl
import matplotlib.pyplot as plt
original_image= X_mnist[36000].reshape(28, 28)
recovered_image= X_mnist_recovered[36000].reshape(28, 28)
plt.subplot(121)
plt.imshow(original_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.subplot(122)
plt.imshow(recovered_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# %%
## 增量 PCA
### PCA的一个问题是它需要处理整个训练集以便 SVD 算法运行。
### 增量 PCA（IPCA）算法可以将训练集分批，一次只对一个批量使用 IPCA 算法。
### 这对大型训练集非常有用，并且可以在线应用 PCA
# 对mnist使用IPCA
from sklearn.decomposition import IncrementalPCA
# 分100批次进行
n_batches=100
inc_pca=IncrementalPCA(n_components=154)
for X_batch in np.array_spplit(X_mnist,n_batches):
    inc_pca.partial_fit(X_batch)
X_mnist_reduced=inc_pca.transform(X_mnist)
# =============================================================================
# 或者调用NumPy 的 memmap 类，该类仅在需要时加载内存中所需的数据
# X_mm=np.memmap(Filename_str,dtype='float32',mode='readonly',shape=(m,n))
# batch_size=m//n_batches
# inc_pca=IncrementalPCA(n_components=154,batch_size=batch_size)
# inc_pca.fit(X_mm)
# =============================================================================

# %%
## 随机 PCA
### SkLearn提供了随机 PCA。这是一种随机算法，可以快速找到前d个主成分的近似值。
from sklearn.decomposition import PCA
rnd_pca=PCA(n_components=154,svd_solver='randomized')
X_reduced=rnd_pca.fit_transform(X_mnist)

# %%
## 非线性降维：核PCA
### 核是一种将实例映射到高维空间的技术,可以执行复杂的非线性投影来降低维度。
# 绘图展示著名的“瑞士卷”数据
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
X, y = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
import matplotlib.pyplot as plt
axes = [-11.5, 14, -2, 23, -12, 15]
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()
from sklearn.decomposition import KernelPCA
# 线性核
lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
# 高斯核
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
# Logistic核
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)
# 绘图展示三种不同的核的投影结果
plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"),
                            (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
plt.show()

# %%
## 寻找最佳PCA超参数：监督方法
# 修改数据以便进行逻辑回归
y= y>6.9
# 定义流水线分类器
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
pipeline_clf= Pipeline([('kpca',KernelPCA(n_components=2)),
               ('logistic',LogisticRegression())])
param_grid= [{'kpca__gamma':np.linspace(0.03,0.05,10),
             'kpca__kernel':['linear','rbf','sigmoid']}]
# 网格搜索
from sklearn.model_selection import GridSearchCV
grid_search= GridSearchCV(pipeline_clf, param_grid, cv=3)
grid_search.fit(X,y)
# 找到最佳
grid_search.best_params_

# %%
## 寻找最佳PCA超参数：非监督方法
# 计算重建前图像误差
rbf_pca = KernelPCA(n_components = 2, kernel="rbf",
                    gamma=0.0433,
                    fit_inverse_transform=True) #重建必需
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
from sklearn.metrics import mean_squared_error
mean_squared_error(X, X_preimage)

### 运用核技巧，使用特征映射 φ 将训练集映射到无限维特征空间，然后使用线性 PCA 将
### 变换的训练集投影。如果在缩减空间中对给定实例实现反向线性 PCA 步骤，则重构点
### 将位于特征空间中，而不是位于原始空间中。由于特征空间是无限维的，我们不能找出
### 重建点，因此我们无法计算真实的重建误差。幸运的是，可以在原始空间中找到一个
### 贴近重建点的点。这被称为重建前图像。一旦你有这个前图像，你就可以测量其与原始
### 实例的平方距离。然后可以选择最小化重建前图像误差的核和超参数。

# %%
## 降维思想：流行学习 降维方法：LLE
### 局部线性嵌入（LLE）是另一种有效的非线性降维方法。LLE 首先测量每个训练实例与其
### 最近邻之间的线性关系，然后寻找能最好地保留这些局部关系的训练集的低维表示
### 这使得它特别擅长展开扭曲的流形，尤其是在没有太多噪音的情况下。（数学原理不懂）
### 这个算法在处理大数据集的时候表现较差。
# LLE降维示例
from sklearn.datasets import make_swiss_roll
X, y = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
from sklearn.manifold import LocallyLinearEmbedding
lle=LocallyLinearEmbedding(n_components=2,n_neighbors=10)
X_reduced=lle.fit_transform(X)
# 绘图展示LLE展开流形效果
import matplotlib.pyplot as plt
plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)
plt.show()

# %%
## 其他降维方法
### 多维缩放（MDS）在尝试保持实例之间距离的同时降低了维度。
### Isomap通过将实例连接至最近的邻居来创建图形，然后在保持实例之间的距离的同时降低维度。
###　t-分布随机邻域嵌入（t-SNE）试图保证相似的实例临近并将不相似的实例分开。它主要
### 用于可视化，尤其是用于可视化高维空间中的实例。
### 线性判别分析（LDA）实际上是一种分类算法，但在训练过程中，它会学习类之间最有区别
### 的轴，然后使用这些轴来定义用于投影数据的超平面。LDA 的好处是投影会尽可能地保持
### 各个类之间距离。所以在运行另一种分类算法（如 SVM）之前，LDA 是很好的降维技术。
# MDS示例
from sklearn.manifold import MDS
mds = MDS(n_components=2, random_state=42)
X_reduced_mds = mds.fit_transform(X)
# Isomap示例
from sklearn.manifold import Isomap
isomap = Isomap(n_components=2)
X_reduced_isomap = isomap.fit_transform(X)
# t-SNE示例
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)
# 绘图展示三种方法的降维效果
plt.figure(figsize=(11,4))
for sub, title, X_reduced in zip((131, 132, 133),
                                  ("MDS", "Isomap", "t-SNE"),
                                  (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
    plt.subplot(sub)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
plt.show()
# LDA示例
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_mnist = mnist["data"]
y_mnist = mnist["target"]
lda.fit(X_mnist, y_mnist)
X_reduced_lda = lda.transform(X_mnist)

# %%







