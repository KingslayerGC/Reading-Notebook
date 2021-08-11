import torch, torchvision

# %%
# 加载预训练模型和数据
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 50, 50)
labels = torch.rand(1, 1000)

# 正向传播计算误差
prediction = model(data) #Resnet18最后是1000fc
loss = (prediction-labels).sum()

# 反向传播计算梯度
loss.backward()

# 梯度下降
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()

# %%
# =============================================================================
# PyTorch中的计算图是动态的，在每个.backward()调用之后，Autograd开始填充新图。可以根
# 据需要在每次迭代中更改形状，大小和操作。
# Autograd跟踪所有requires_grad设置为True的张量。对于不需要梯度的张量，将此属性设置
# 为False会将其从梯度计算中排除。
# =============================================================================
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)
a = x + y
print(f"Does a require gradients? : {a.requires_grad}")
b = x + z
print(f"Does b require gradients?: {b.requires_grad}")

# 冻结所有参数
from torch import nn
for param in model.parameters():
    param.requires_grad = False

# 只训练fc层
model.fc = nn.Linear(512, 10)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()