# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 20:31
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 7.合并与切割.py
# @Software: PyCharm
import torch

'''
    合并
'''
# cat()：拼接函数（拼接的是维度）
class1 = torch.rand(4, 32, 9)
class2 = torch.rand(4, 32, 9)
print(torch.cat([class1,class2],dim=0).shape) # torch.Size([7, 32, 8])

# stack()：拼接函数，拼接后会产生新的维度,必须保证所拼接的维度的大小相同
print(torch.stack([class1,class2],dim=0).shape) # torch.Size([2, 4, 32, 9])

'''
    拆分
'''
# split()：根据数量进行拆分
class3 = torch.rand(8,32,8)
split1,split2,split3 = class3.split([3, 4, 1], dim=0)
print(split1.shape,split2.shape,split2.shape) # torch.Size([3, 32, 8]) torch.Size([4, 32, 8]) torch.Size([4, 32, 8])
split4,split5 = class3.split(4, dim=0)
print(split4.shape,split5.shape) # torch.Size([4, 32, 8]) torch.Size([4, 32, 8])

# chunk()：按照长度进行划分
rand = torch.rand(8, 32, 8)
chunk1,chunk2 = rand.chunk(2,dim=0) # 一共分为两份
print(chunk1.shape,chunk2.shape) # orch.Size([4, 32, 8]) torch.Size([4, 32, 8])
