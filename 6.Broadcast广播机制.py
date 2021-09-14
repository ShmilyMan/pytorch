# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 19:54
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 6.Broadcast广播机制.py
# @Software: PyCharm
import torch
rand = torch.rand(3, 3)
print(rand)
tensor = torch.tensor([1])
print(tensor)
print(rand + tensor)
'''
tensor([[1.6023, 1.9277, 1.7125],
        [1.6171, 1.0831, 1.4861],
        [1.2471, 1.7562, 1.2125]])
'''
