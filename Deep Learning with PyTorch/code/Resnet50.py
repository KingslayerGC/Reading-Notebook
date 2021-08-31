# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.transforms.functional import rotate
from torchvision.transforms import Compose, ToTensor, Resize,\
    RandomHorizontalFlip, RandomVerticalFlip, Lambda, ColorJitter

# 随机旋转
def random_rot90(image):
    random = np.random.uniform()
    if random > 0.75:
        return rotate(image, 90)
    elif random > 0.5:
        return rotate(image, 180)
    elif random > 0.25:
        return rotate(image, 270)
    else:
        return image

# 自定义数据集
class MyDataset(Dataset):
    
    def __init__(self, image_dir, images, labels, image_dim, read_before, augment):
        self.image_dir = image_dir
        self.images = images
        self.labels = labels
        self.read_before = read_before
        if augment:
            self.transfrom = Compose([ToTensor(),
                                      Resize(image_dim),
                                      RandomHorizontalFlip(),
                                      RandomVerticalFlip(),
                                      Lambda(lambda x : x if np.random.uniform()>0.5 else x.transpose(1,2)),
                                      Lambda(random_rot90),
                                      ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                                      ])
        else:
            self.transfrom = Compose([ToTensor(),
                                      Resize(image_dim),
                                      ])
    
    def __getitem__(self, index):
        if self.read_before:
            # copy防止augment破坏原数据
            image = self.images[index].copy()
        else:
            image = plt.imread(self.image_dir + "/" + self.images[index])
        return self.transfrom(image), self.labels[index]
    
    def __len__(self):
        return len(self.labels)

# 分割训练集和测试集
def split_set(df, col, train_size, test_size):
    split = StratifiedShuffleSplit(train_size=train_size, test_size=test_size, random_state=42)
    for ind1, ind2 in split.split(df, df[col]):
        return df.iloc[ind1], df.iloc[ind2]

# 降采样
def subsample(df, col):
    count = df[col].value_counts()
    size = count.values.min()
    df_list = []
    for i in count.index:
        df_list.append(df[df[col]==i].sample(size))
    return pd.concat(df_list)


def get_dataloader(image_dir, label_path, train_size, test_size, batch_size, image_dim,
                   sub, augment, read_before):
    
    data = pd.read_csv(label_path)
    
    # 是否降采样
    if sub:
        data = subsample(data, 'label')    
    
    # 分割训练集和测试集
    count = (data['label'].value_counts()/len(data)).sort_index().values
    train_set, test_set = split_set(data, 'label', train_size, test_size)
    X_train, y_train = train_set['image_id'].values, train_set['label'].values
    X_test, y_test = test_set['image_id'].values, test_set['label'].values
    
    # 是否预先读取图片
    if read_before:
        for array in [X_train, X_test]:
            for i in range(len(array)):
                array[i] = plt.imread(image_dir + "/" + array[i])
    
    # 生成数据集
    train_ds = MyDataset(image_dir, X_train, y_train, image_dim, read_before, augment)
    test_ds = MyDataset(image_dir, X_test, y_test, image_dim, read_before, augment=False)
    
    # 返回小批次迭代器
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size*2), 1/count

# %%
# =============================================================================
# 准备卷积网络
# =============================================================================
from torch import nn, optim

class Lambda2(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Conv(nn.Module):
    
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding='same', _bn_relu=True, groups=1):
        super().__init__()
        if stride != 1:
            padding = kernel_size//2
        # 旧版本pytorch不能使用'same'作为padding参数，可能报错
        conv = nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, groups=groups)
        nn.init.kaiming_normal_(conv.weight)
        # 是否使用BN和激活函数
        if _bn_relu:
            self.model = nn.Sequential(conv, nn.BatchNorm2d(channel_out), nn.ReLU())
        else:
            self.model = conv

    def forward(self, xb):
        return self.model(xb)

class BottleNeck(nn.Module):
    
    def __init__(self, channel_in, channel_out, groups):
        super().__init__()
        if channel_out//channel_in == 2:
            stride = 2
        else:
            stride = 1
        if channel_out == channel_in:
            self.shortcut = None
        else:
            self.shortcut = Conv(channel_in, channel_out, 1, stride, _bn_relu=False)
        
        if groups == 1:
            channel_mid = channel_out//4
        else:
            channel_mid = channel_out//2
        
        # 先进行预激活
        bn_relu = nn.Sequential(nn.BatchNorm2d(channel_in), nn.ReLU())
        conv1 = Conv(channel_in, channel_mid, 1, stride)
        conv2 = Conv(channel_mid, channel_mid, 3, groups=groups)
        conv3 = Conv(channel_mid, channel_out, 1, _bn_relu=False)
        self.residual = nn.Sequential(bn_relu, conv1, conv2, conv3)
        
    def forward(self, xb):
        if self.shortcut == None:
            return xb + self.residual(xb)
        else:
            return self.shortcut(xb) + self.residual(xb)

class Resnet50(nn.Module):
    
    def __init__(self, n_output, groups=1):
        super().__init__()
        respart = []
        for block in [(64,256,3), (256,512,4), (512,1024,6), (1024,2048,3)]:
            for i in range(block[2]):
                if i < 1:
                    respart.append(BottleNeck(block[0], block[1], groups))
                else:
                    respart.append(BottleNeck(block[1], block[1], groups))
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(3, affine=False), #输入标准化
            Conv(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            *respart,
            nn.AdaptiveAvgPool2d(1), #全局平均池化
            Lambda2(lambda x: x.view(-1, 2048)),
            nn.Linear(2048, n_output), #FC层
            nn.BatchNorm1d(n_output)
            )
        
    def forward(self, xb):
        return self.model(xb)

# %%
# =============================================================================
# 训练并得到结果
# =============================================================================
import torch.nn.functional as F

def focal_loss(logits, labels):
    proba = F.softmax(logits, dim=1)
    pt = proba.gather(dim=1, index=labels.view(-1,1)).view(-1)
    alpha = torch.tensor(ALPHA, device=DEV).gather(dim=0, index=labels)
    return sum(-alpha * (1.0-pt)**GAMMA * pt.log())

def acc_batch(model, loss_func, xb, yb, dev, opt=None):
    # 把数据加载到指定位置上
    xb, yb = xb.to(dev), yb.to(dev)
    # 计算各类得分
    logits = model(xb)
    # 如果是训练模式
    if opt != None:
        # 计算误差
        loss = loss_func(logits, yb)
        # 反向传播计算梯度
        loss.backward()
        # 梯度下降
        opt.step()
        # 梯度清零
        opt.zero_grad()
    return sum(torch.argmax(logits, dim=1)==yb), len(yb)

def fit(epochs, model, loss_func, opt, train_dl, test_dl, dev):
    for epoch in range(epochs):
        # 开启训练模式
        model.train()
        # 训练
        for xb, yb in train_dl:
            acc_batch(model, loss_func, xb, yb, dev, opt)
        # 关闭训练模式
        model.eval()
        # 冻结参数（不计算梯度）方便计算准确率
        with torch.no_grad():
            print('epoch', epoch + 1)
            acc_num, num = zip(
                *[acc_batch(model, loss_func, xb, yb, dev) for xb, yb in train_dl])
            print('train accuracy: {:.3f}'.format(sum(acc_num)/sum(num)))
            acc_num, num = zip(
                *[acc_batch(model, loss_func, xb, yb, dev) for xb, yb in test_dl])
            print('test accuracy: {:.3f}'.format(sum(acc_num)/sum(num)))

# %%
# =============================================================================
# 主程序
# =============================================================================
# 全局变量
IMG_PATH = r"D:\kaggle_leafdisease\train_images"
LABEL_PATH = r"D:\kaggle_leafdisease\train.csv"
IMAGE_DIM = (512, 512)
BATCH_SIZE = 4
GAMMA = 2
DEV = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
DEV = torch.device("cpu")

# 获得数据迭代器
train_dl, test_dl, ALPHA = get_dataloader(IMG_PATH, LABEL_PATH, train_size=160, test_size=48, batch_size=BATCH_SIZE, image_dim=IMAGE_DIM,
                                          sub=False, augment=True, read_before=True)

# 建立模型
model = Resnet50(n_output=5, groups=32).to(DEV)

# 训练并得到结果
opt = optim.Adam(model.parameters())
#fit(10, model, focal_loss, opt, train_dl, test_dl, DEV)

# %%
# =============================================================================
# 可视化
# =============================================================================
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 生成储存数据的文件夹
writer = SummaryWriter(r"C:\Users\KingslayerGC\runs\experiment_1")

# 获得一些图片
dataiter = iter(train_dl)
images, labels = dataiter.next()

# 展示图片
img_grid = torchvision.utils.make_grid(images)
writer.add_image('train_images', img_grid)

# 展示模型
writer.add_graph(model, images)

# 投影（默认PCA）
writer.add_embedding(images.view(-1, 512*512*3),
                     metadata=labels,
                    label_img=images)

# 添加标量，将自动绘制折线图
# writer.add_scalar

# 添加评估指标
# writer.add_pr_curve

writer.close()
