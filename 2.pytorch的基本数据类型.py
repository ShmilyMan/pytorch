# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 22:08
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 2.pytorch的基本数据类型.py
# @Software: PyCharm

import torch
import numpy as np

a = torch.randn(2, 3) # 随机生成服从N(0,1)的随机的两行三列的矩阵
'''
tensor([[-0.0021, -0.4088, -1.2898],
        [ 0.3904, -0.3344,  1.0378]])
'''
print(a)

'''
    1.检查pytorch的数据类型：
        Pytorch数据类型的检查可以通过三个方式：
        1）python内置函数type()
        2）Tensor的成员函数Tensor.type()
        3）Pytorch提供的工具函数isinstance()
'''
print(a.type()) # torch.FloatTensor
print(type(a)) # <class 'torch.Tensor'> 不能确定具体的数据类型
print(isinstance(a,torch.FloatTensor)) # True 一般用于判断某个变量是否为具体的数据类型

'''
    2.数据类型转换：
        1）Tensor类型的变量直接调用long(), int(), double(),float(),byte()等函数就能将Tensor进行类型转换；
        2）在Tensor成员函数type()中直接传入要转换的数据类型。
        当你不知道要转换为什么类型时，但需要求a1,a2两个张量的乘积，可以使用a1.type_as(a2)将a1转换为a2同类型。
'''
print('----------------------------')
b = a.type(torch.IntTensor)
print(b.type()) # torch.IntTensor
c = a.int()
print(c.type()) # torch.LongTensor
a = a.type_as(c) # 必须对a重新赋值
print(a.type()) # torch.FloatTensor

'''
    3.Tensor和numpy.ndarray之间还可以相互转换，其方式如下：
        1）Numpy转化为Tensor：torch.from_numpy(numpy矩阵)
        2）Tensor转化为numpy：Tensor矩阵.numpy()
'''
# Tensor转化为numpy
randn = torch.randn(3, 2)
randn_numpy = randn.numpy()
'''
tensor([[ 0.6458, -0.6467],
        [-0.2682,  0.6706],
        [ 1.6167,  0.4424]])
'''
print(randn)
'''
[[ 0.6457594  -0.6467263 ]
 [-0.26822838  0.6706486 ]
 [ 1.6167423   0.44236663]]
'''
print(randn_numpy)

# Tensor转化为numpy
array = np.array([[2, 3], [3, 4], [7, 8]])
from_numpy_torch = torch.from_numpy(array)
'''
[[2 3]
 [3 4]
 [7 8]]
'''
print(array)
'''
tensor([[2, 3],
        [3, 4],
        [7, 8]], dtype=torch.int32)
'''
print(from_numpy_torch)

print('------------------------')
'''
    4.创建一个深度为0的张量
'''
tensor = torch.tensor(1.1)
print(tensor) # tensor(1.1000)
# 获取张量的类型
print(tensor.size()) # torch.Size([])
print(tensor.shape) # torch.Size([])
print(len(tensor.size())) # 0

'''
    5.创建一个一维的张量
'''
torch_tensor = torch.tensor([1.1, 2.2])
print(torch_tensor) # tensor([1.1000, 2.2000])
print(torch_tensor.size()) # torch.Size([2])

# 按照正态分布随机产生
float_tensor = torch.FloatTensor(2)
print(float_tensor) # tensor([-1.0842e-19,  1.8875e+00])

ones = torch.ones(2)
print(ones.int())

'''
    6.创建一个三维的张量
'''
rand = torch.rand(1, 2, 3) # 0,1的均匀分布产生的
print(rand)
print(rand.shape) # torch.Size([1, 2, 3])
print(list(rand.size())) # [1, 2, 3]

'''
    7.创建一个四维的张量
'''
torch_rand = torch.rand(2, 3, 28, 28)
print(torch_rand)

# 所占内存大小
print(torch_rand.numel()) # 4704     2 * 3 * 28 * 28

# 深度（维度）
print(torch_rand.dim()) # 4
