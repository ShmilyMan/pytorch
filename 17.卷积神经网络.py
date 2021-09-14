# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 16:42
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 17.卷积神经网络.py
# @Software: PyCharm
import torch
import torch.nn as nn

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
x = torch.randn(1,1,28,28)
out = layer(x)

print(out.shape)
print(out)
