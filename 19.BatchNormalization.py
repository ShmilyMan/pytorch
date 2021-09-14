# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 19:45
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 19.BatchNormalization.py
# @Software: PyCharm
import torch
import torch.nn as nn

'''1.对1维的向量'''
x = torch.rand(100, 16, 784)
layer = nn.BatchNorm1d(16)
out = layer(x)
# 运行时的均值
print(layer.running_mean)
# 运行时的方差
print(layer.running_var)

'''2.对2维的向量'''
x = torch.rand(1,16,7,7)
layer = nn.BatchNorm2d(16)
out = layer(x)
print(out.shape)
# 新计算的均值
print(layer.weight)
# 新计算的方差
print(layer.bias)

print(vars(layer))
