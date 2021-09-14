# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 23:18
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 16.数据集的测试.py
# @Software: PyCharm
import torch
from torch.nn import functional as F

data = torch.rand(4, 10)
pred = F.softmax(data, dim=1)
label_pred = pred.argmax(dim=1)
print(label_pred)
print(data.argmax(dim=1))

label_test = torch.tensor([2, 2, 2, 2])
result = torch.eq(label_pred, label_test)
item = result.sum().float().item()
print(item / 4)


import torch.utils.data as Data



BATCH_SIZE = 3 #批训练数据个数



x = torch.linspace(1,10,10)  #x data (torch tensor)

y = torch.linspace(10,1,10)  #y data (torch tensor)



#随后我们需要把X和Y组成一个完整的数据集，并转化为pytorch能识别的数据集类型：



torch_dataset = Data.TensorDataset(x,y)

#可以看出我们把X和Y通过Data.TensorDataset() 这个函数拼装成了一个数据集，数据集的类型是【TensorDataset】。



# 把 dataset 放入 DataLoader

loader = Data.DataLoader(

        dataset = torch_dataset,

        batch_size = BATCH_SIZE,

        shuffle = True,

        num_workers = 0,

        )



for epoch in range(5): #训练所有数据5次

    i = 0

    for batch_x, batch_y in loader:

        i = i + 1

        print('Epoch:{} | num:{} | batch_x:{} | batch_y:{}'

              .format(epoch,i,batch_x,batch_y))