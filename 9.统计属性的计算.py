# -*- coding: utf-8 -*-
# @Time    : 2020/12/2 15:22
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 9.统计属性的计算.py
# @Software: PyCharm
import torch

'''1.norm()：求范数'''
a = torch.full([8], 1)
b = a.view(2,4).float()
c = a.view(2,2,2).float()
# 求1范数
print(b.norm(1)) # tensor(8.)
print(c.norm(1)) # tensor(8.)

# 求2范数
print(b.norm(2)) # tensor(2.8284)
print(c.norm(2)) # tensor(2.8284)

# 在指定的维度上求2范数
print(b.norm(2,dim=1))
print(c.norm(2,dim=0))

'''2.mean()：求平均值,max():求最大值,min():求最小值,sum()：求和,prod()：求所有元素的乘积'''
num_lst = torch.arange(0, 10).float()
# tensor(9.) tensor(0.) tensor(45.) tensor(0.) tensor(4.5000)
print(num_lst.max(),num_lst.min(),num_lst.sum(),num_lst.prod(),num_lst.mean())

'''3.求最小值和最大值的索引'''
num = torch.rand(4, 10)
# 先将二维的打成一维的，再计算索引
print(num.argmax()) # tensor(38)
print(num.argmin()) # tensor(39)
# 在每一个维度上分别计算索引
print(num)
print(num.argmin(dim=1)) # tensor([6, 9, 2, 8])
print(num.min(dim=1))

# 使得计算的结果的维度为之前的维度不变
print(num.min(dim=1,keepdim=True))

'''4.topk():求满足最大值或最小值的前几项'''
print(num.topk(3,dim=1)) # 求的是满足最大值的前三项
print(num.topk(3,dim=1,largest=False)) # 求满足最小值的前三项

'''5.kthvalue()：求满足第几小的数的值'''
print(num.kthvalue(8,dim=1))

'''6.矩阵的比较操作'''
a = torch.rand(4,10)
b = torch.rand(4,10)
print(a > 0.5)
print(torch.eq(a,b))
print(torch.equal(a,b))

