# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 21:22
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 8.基本运算.py
# @Software: PyCharm
import torch
'''1.矩阵的加减乘除运算'''
rand1 = torch.ones(3, 3)
rand2 = torch.ones(3)
print(rand1 + rand2)
print(rand1 - rand2)
print(rand1 * rand2) # 矩阵对应的元素进行乘法运算
print(rand1 / rand2) # 矩阵对应的元素进行除法运算

'''2.矩阵相乘'''
rand3 = torch.ones(3, 4)
rand4 = torch.ones(4, 3)
print(torch.matmul(rand3,rand4)) # 矩阵的乘法

'''3.矩阵的乘方'''
print(rand1 ** 2)

'''4.矩阵的开方'''
print(rand1 ** (0.5))
print(rand1.rsqrt()) # 矩阵开方求导

'''5.求矩阵的指数，对数'''
a = rand1.exp()
print(rand1.exp())
print(a.log())

'''6.估计值，近似值'''
value = torch.tensor(3.14)
print(value) # tensor(3.1400)
print(value.floor()) # tensor(3.)
print(value.ceil()) # tensor(4.)
print(value.round()) # tensor(3.)  四舍五入
print(value.trunc()) # tensor(3.) 取整数部分
print(value.frac()) # tensor(0.1400) 取小数部分

'''7.clamp():限制值的范围'''
randn5 = torch.randn(3, 4) * 15
print(randn5)
print(randn5.clamp(0,10))

