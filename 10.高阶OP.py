# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 18:26
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 10.高阶OP.py
# @Software: PyCharm
import torch

'''1.where()'''
a = torch.rand(3, 3)
b = torch.ones(3, 3)
c = torch.zeros(3, 3)
print(a)
print(b)
print(c)
print(torch.where(a > 0.5,b,c)) # 使用b,c对a重新进行赋值

'''2.gather():按一定条件将一个矩阵映射为另一个矩阵'''
print('---------------------')
a = torch.rand(4, 10)
result = a.topk(3, 1)
index = result[1]
print(index)
templet = torch.arange(0, 10) + 100
print(templet)
print(torch.gather(templet.expand(4,10),dim=1,index=index))
