# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 16:05
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 13.链式法则.py
# @Software: PyCharm
'''x[w1,b1]-> y1 [w2,b2] -> y2'''
import torch
x = torch.tensor(1.)
w1 = torch.tensor(2.,requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2.,requires_grad=True)
b2 = torch.tensor(1.)

y1 = w1 * x + b1
y2 = w2 * y1 + b2
y2_y1 = torch.autograd.grad(y2,[y1],retain_graph=True)[0]
y1_w1 = torch.autograd.grad(y1,[w1],retain_graph=True)[0]
y2_w1 = torch.autograd.grad(y2,[w1],retain_graph=True)[0]
print(y2_y1 * y1_w1,y2_w1)