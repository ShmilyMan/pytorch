# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 9:56
# @Author  : Wxl
# @Email   : 154831156@qq.com
# @File    : 1.梯度下降算法案例.py
# @Software: PyCharm
import numpy as np
import torch

points = np.genfromtxt('data.csv', delimiter=',')
# y = wx + b
# (wx + b - y) ** 2
'''
    计算给定的w,b的误差
'''
def complate_error_give_for_points(w,b,points):
    error = 0
    for temp in points:
        error += (w * temp[0] + b - temp[1]) ** 2

    return error / float(len(points))

'''
    计算每一个给定的坐标点所对应的w,b的梯度差
'''
def complate_gray_points(w,b,points,lr):
    w_gray = 0
    b_gray = 0
    length = len(points)
    for temp in points:
        w_gray += -(2/length) * temp[0] * (temp[1] - (w * temp[0] + b))
        b_gray += -(2 / length) * (temp[1] - (w * temp[0] + b))
    new_b = b - (lr * b_gray)
    new_w = w - (lr * w_gray)
    return new_b,new_w

'''
    迭代的计算梯度差，这样越来越接近他的最小值
'''
def run_complate_gray(b,w,points,lr,iterations):
    for temp in range(iterations):
        b = complate_gray_points(w,b,points,lr)[0]
        w = complate_gray_points(w,b,points,lr)[1]
    return b,w

def run():
    b = 0 # b的初始值
    w = 0 # w的初始值
    lr = 0.0001 # 学习速度
    iterations = 10000 # 迭代的次数
    print(f'开始时w = {w},b = {b},误差为：{complate_error_give_for_points(w,b,points)}')
    print('Runing...')
    b = run_complate_gray(b,w,points,lr,iterations)[0]
    w = run_complate_gray(b,w,points,lr,iterations)[1]
    print(f'结束时w = {w},b = {b},误差为：{complate_error_give_for_points(w,b,points)}')


if __name__ == '__main__':
    run()