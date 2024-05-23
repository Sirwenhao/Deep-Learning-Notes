# 24/2/27 author:WH
# 函数用法

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# 创建一个包含多个图像的三维张量
images = torch.rand(16, 3, 64, 64) # 16张RGB图像，每张大小为64x64

# 使用make_grid()创建图像网格
grid_image = make_grid(images, nrow=4, padding=2, normalize=True)

# 将PyTorch张量转换为NumPy数组，并调整通道顺序
grid_image_np = grid_image.permute(1, 2, 0).numpy()

# 显示图像网格
plt.imshow(grid_image_np)
plt.axis('off')
plt.show()