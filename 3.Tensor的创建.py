# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 13:08
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 3.Tensor的创建.py
# @Software: PyCharm
import numpy as np
import torch
'''1.从numpy中创建'''
array = np.array([2.2, 3.3])
print(torch.from_numpy(array)) # tensor([2.2000, 3.3000], dtype=torch.float64)

'''2.直接使用torch进行创建'''
# 给定值
tensor = torch.tensor([1.1, 2.2])
print(tensor) #  tensor([1.1000, 2.2000])

# 给定维度
print(torch.Tensor(2,3))

'''3.生成随机的Tensor'''
print(torch.empty(2,3))
'''
tensor([[0., 0., 0.],
        [0., 0., 0.]])
'''
print(torch.FloatTensor(2,3))
'''
tensor([[ 0.0000e+00,  0.0000e+00,  2.1019e-44],
        [ 0.0000e+00, -6.3484e-38,  7.2727e-43]])
'''
print('----------------------')
# 均匀分布
a = torch.rand(2,3)
print(a)
print(torch.rand_like(a))
# [min,max),shape
print(torch.randint(1,10,[2,3]))
'''
tensor([[8, 9, 1],
        [5, 7, 5]])
'''

# 正态分布
print(torch.randn(2,3))

'''4.设置默认的Tensor类型'''
print(torch.tensor([1.1,2.2]).type()) # torch.FloatTensor
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.1,2.2]).type()) # DoubleTensor

'''5.full函数'''
print(torch.full([],2)) # tensor(2)
print(torch.full([2,3],1)) # tensor([[1, 1, 1],
                                #   [1, 1, 1]])
'''6.arange函数'''
print(torch.arange(0,10)) # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(torch.arange(0,10,2)) # tensor([0, 2, 4, 6, 8])

'''7.linspace,logspace函数'''
print(torch.linspace(0,10,steps=4)) # tensor([ 0.0000,  3.3333,  6.6667, 10.0000])
print(torch.logspace(0,1,10)) # tensor([ 1.0000,  1.2915,  1.6681,  2.1544,  2.7826,  3.5938,  4.6416,  5.9948,7.7426, 10.0000])

'''8.ones,zeros,eye'''
print(torch.ones(3,3))
print(torch.eye(3,3))
print(torch.zeros(3,3))

'''9.randperm函数'''
print(torch.randperm(10)) # tensor([6, 9, 0, 5, 2, 4, 1, 3, 7, 8])
b = torch.rand(2, 3)
print(b)
print(b[0])
