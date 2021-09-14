# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 17:07
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 18.Pooling&upsample.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.nn import functional as F
'''1.池化层下采样'''
x = torch.randn(1, 1, 28, 28)
layer = nn.MaxPool2d(2, 2)
out = layer(x)
print(out.shape)

'''2.上采样'''
out2 = F.interpolate(out, scale_factor=2, mode='nearest')
print(out2.shape)

'''3.ReLU'''
layer = nn.ReLU(inplace=True)
print(layer(out).shape) # 把负数的像素点去掉了，使图像更平滑
