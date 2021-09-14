# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 21:14
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 11.激活函数与Loss梯度.py
# @Software: PyCharm
import torch
from torch.nn import functional as F

'''1.sigmoid激活函数，范围：0~1'''
a = torch.linspace(-100,100,10)
print(a)
sigmoid = torch.sigmoid(a)
print(sigmoid) # 当x趋向于负无穷的时候，该函数趋向于零，当x趋向于正无穷的时候，该函数趋向于1

'''2.tanh激活函数，范围：-1~1'''
a = torch.linspace(-1,1,10)
tanh = torch.tanh(a)
print(tanh) # 当x趋向于负无穷时该函数趋向于-1，当x趋向于正无穷时，该函数趋向于1

'''3.loss=sum{[y-(xw+b)]^2}:错误准则函数'''
x = torch.ones(1)
w = torch.full([1],2).float()
grad_ = w.requires_grad_()
loss = F.mse_loss(x * w, torch.ones(1))

# 求梯度
gradw = torch.autograd.grad(loss,[w])
print(gradw)

# 求梯度的另一种方法
x = torch.ones(1)
w = torch.full([1],2).float()
grad_ = w.requires_grad_()
loss = F.mse_loss(x * w, torch.ones(1))
loss.backward()
print(w.grad)

'''4.softmax函数，将输入的值转化成0~1上的值，而且这些值相加为1，可以作为概率进行处理'''
a = torch.rand(3)
a.requires_grad_()
softmax = F.softmax(a, dim=0)
# 求偏导
print(torch.autograd.grad(softmax[0], [a]))
softmax = F.softmax(a, dim=0)
print(torch.autograd.grad(softmax[1], [a]))
softmax = F.softmax(a, dim=0)
print(torch.autograd.grad(softmax[2], [a]))