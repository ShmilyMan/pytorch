# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 16:50
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 4.索引与切片.py
# @Software: PyCharm

import torch

picture = torch.randn(4, 3, 28, 28)
print(picture.size()) # torch.Size([4, 3, 28, 28])
# 1.切片操作
print(picture[0:1,0:1].size()) # torch.Size([1, 1, 28, 28])

# 2.索引操作 获取具体位置上的对象
print(picture.index_select(0,torch.tensor([0,1])).size())
