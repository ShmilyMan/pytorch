# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 19:13
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 5.维度变换.py
# @Software: PyCharm
import torch
'''1.view():进行维度的变换，可以进行扩维度、和缩维'''
picture = torch.randn(4, 1, 28, 28)
a = picture.view(4,28 * 28)
# print(a) # 代表4个图片，每个图片为一个一维的向量

# 还原成原来的维度   注意：必须与变换之前的维度信息保持一致
# print(a.view(4,1,28,28))

'''2.unsqueeze():增加维度  squeeze():缩减维度,只能去掉为size为1的维度,如果不给定参数，则去掉全部能去掉的 '''
randn = torch.randn(32)
b = randn.unsqueeze(0).unsqueeze(2).unsqueeze(3)
print(b.size()) # torch.Size([1, 32, 1, 1])
print(b.squeeze().size()) # 去掉所有size为1的维度  torch.Size([32])
print(b.squeeze(0).size()) # torch.Size([32, 1, 1])

'''3.expand():维度的扩展'''
c = torch.randn(1,32,1,1)
print(c.expand(4,32,14,14))

'''4.二维矩阵的转置'''
rand = torch.rand(3, 3)
rand_t = rand.t()
print(rand)
print(rand_t)

'''5.transpose():改变维度的位置，改变后数据发生改变，如果要还原原来的数据类型，需要对改变后的数据进行连续化'''
torch_randn = torch.randn(4, 3, 32, 32)
transpose = torch_randn.transpose(0, 2)
print(transpose.shape) # torch.Size([32, 3, 4, 32])

# 还原为原来的数据
view__transpose = transpose.contiguous().view(32, 3 * 4 * 32).view(32, 3, 4, 32).transpose(0, 2)
print(view__transpose.shape) # torch.Size([4, 3, 32, 32])

'''6.permute()：改变矩阵的维度的位置，一次改变多个位置（多个位置随意改变）'''
torch_rand = torch.rand(4, 3, 28, 28)
print(torch_rand.permute(3,2,0,1).shape) # torch.Size([28, 28, 4, 3])
