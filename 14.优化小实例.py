# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 17:00
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 14.优化小实例.py
# @Software: PyCharm
import torch

# f(x,y) = (x^2+y-11)^2+(x+y^2-7)^2
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

# 优化方法
x = torch.tensor([4., 0.], requires_grad=True)
optim_adam = torch.optim.Adam([x], lr=1e-3)

for step in range(20000):
    pred = himmelblau(x)
    optim_adam.zero_grad()
    pred.backward()

    # x' = x - lr * delta
    # y' = y - lr * delta
    optim_adam.step()
    if step % 2000 == 0:
        print(f'坐标为：{x.tolist()},函数值值为：{pred}')



