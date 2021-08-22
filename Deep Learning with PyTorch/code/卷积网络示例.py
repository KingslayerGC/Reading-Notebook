# %%
# =============================================================================
# 准备Minst数据和GPU
# =============================================================================
import torch
import numpy as np
from sklearn.datasets import fetch_openml
from torch.utils.data import TensorDataset, DataLoader

# 下载Minst数据
mnist = fetch_openml('mnist_784')
X, y = mnist["data"].values, mnist["target"].astype(int).values

def get_dataloader(X, y, train_size=0.8, batch_size=16):
    # 打乱数据
    shuffle_index = np.random.permutation(len(X))
    X, y = map(torch.tensor, (X[shuffle_index], y[shuffle_index]))
    X, y = X.to(torch.float32).view(-1, 1, 28, 28
                                    ).to(dev), y.to(torch.long).to(dev)
    # 按比例分割训练集和测试集
    index = int(len(X)*train_size)
    train_ds = TensorDataset(X[:index], y[:index])
    valid_ds = TensorDataset(X[index:], y[index:])
    # 返回一个生成小批次的迭代器
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True),\
        DataLoader(valid_ds, batch_size=batch_size*2)

# %%
# =============================================================================
# 准备卷积网络
# =============================================================================
from torch import nn, optim
import torch.nn.functional as F

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.adaptive_avg_pool2d(xb, 1)
        # 输出（size, 10）
        return xb.view(-1, xb.size(1))

# %%
# =============================================================================
# 训练并得到结果
# =============================================================================
def loss_batch(model, loss_func, xb, yb, opt=None):
    score = model(xb)
    # 计算误差
    loss = loss_func(score, yb)
    # 如果是训练模式
    if opt != None:
        # 反向传播计算梯度
        loss.backward()
        # 梯度下降
        opt.step()
        # 梯度清零
        opt.zero_grad()
    return sum(torch.argmax(score, dim=1)==yb), len(yb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # 开启训练模式
        model.train()
        # 训练
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        # 关闭训练模式
        model.eval()
        # 冻结参数（不计算梯度）方便计算准确率
        with torch.no_grad():
            print('epoch', epoch + 1)
            acc_num, num = loss_batch(model, loss_func, xb, yb)
            print('batch accuracy: {:.3f}'.format(acc_num/num))
            acc_num, num = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            print('test accuracy: {:.3f}'.format(sum(acc_num)/sum(num)))

# %%
# =============================================================================
# 主程序
# =============================================================================
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

train_dl, valid_dl = get_dataloader(X, y, 0.8, 64)

model = Mnist_CNN().to(dev)
opt = optim.Adam(model.parameters()) #使用SGD容易陷入局部最小

fit(10, model, F.cross_entropy, opt, train_dl, valid_dl)

