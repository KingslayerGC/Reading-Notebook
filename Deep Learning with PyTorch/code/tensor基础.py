import torch
import numpy as np

# %%
# =============================================================================
# 生成
# =============================================================================
# 直接转换
l = [[1,2], [3,4]]
array = np.array(l)
tensor = torch.tensor(l)
torch.tensor(array)

# 继承
x_ones = torch.ones_like(tensor)
x_rand = torch.rand_like(tensor, dtype=torch.float)
print(x_ones)
print(x_rand)

# 用维度生成
shape = (2,3,)
torch.rand(shape)
torch.ones(shape)
torch.zeros(shape)

# %%
# =============================================================================
# 属性和方法
# =============================================================================
# 打印属性
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 导入GPU
tensor = torch.ones(4, 4)
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

# 切片
tensor[:,1] = 0

# 拼接
torch.cat([tensor, tensor, tensor], dim=1)

# 点乘
tensor.mul(tensor)
tensor * tensor

# 矩阵乘法
tensor.matmul(tensor)
tensor @ tensor.T

# 赋值加法
tensor.add_(5)

# 与array共享内存(仅限CPU)
tensor = tensor.to('cpu')
tensor.numpy()
array = np.ones(5)
torch.from_numpy(array)