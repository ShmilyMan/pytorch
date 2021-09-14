# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 23:16
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 12.感知机的梯度推导.py
# @Software: PyCharm
import torch
from torch.nn import functional as F

'''1.单一感知机的推导'''
x = torch.randn(1,10)
w = torch.randn(1,10,requires_grad=True)
temp = torch.matmul(x, w.t())
# 通过sigmoid函数
result = torch.sigmoid(temp)
# 求偏导
loss = F.mse_loss(result, torch.ones(1,1))
loss.backward()
print(w.grad)

'''2.多感知机的推导(第二层为两个节点)'''
x = torch.randn(1,10)
w = torch.randn(2,10,requires_grad=True)
result = torch.sigmoid(torch.matmul(x, w.t()))
loss = F.mse_loss(result,torch.ones(1,2))
loss.backward()
print(w.grad)


