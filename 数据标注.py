# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/01/07 12:51
# @Author  : Zhangkai
# @Version : python3.9
# @Desc    : $END$

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from time import time
from skimage.io import imread,imshow
from PIL import Image
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, square, diamond
from skimage.color import rgb2gray
import math
from skimage.util import img_as_ubyte
from imageio import imsave
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def get_pre_pic(pre_label, colors_, shapess):
    # colors_ = cp.array(colors_)
    print(list(set(pre_label.flatten())))
    pre_label = pre_label + 1
    pic = np.zeros((shapess[0], shapess[1], 3), dtype=np.uint8)  # 创建三通道图像
    for i in range(shapess[0]):
        if i % 100 == 0:  print(i, end=' ')
        for j in range(shapess[1]):
            if pre_label[i, j] != -1:
                pic[i, j, 0] = colors_[colors_[:, 0] == pre_label[i, j]][:, 3]
                pic[i, j, 1] = colors_[colors_[:, 0] == pre_label[i, j]][:, 2]
                pic[i, j, 2] = colors_[colors_[:, 0] == pre_label[i, j]][:, 1]
            else:
                pic[i, j, :] = [0,0,0]
                # cv2.imwrite('D:\\temp_data\\XiongAn\\' + cc + str(x) + 'pre_.png', pic)  # cv2输出路径必须是英文
    print()
    return np.array(pic)
def get_index_pic():
    filename = 'D:\A_real_data\HSI\coordinate_新_周恒\\'
    # dataList = ["Indian Pines", "salinas", "Pavia University", "HanChuan", "HongHu", "LongKou", ]
    dataList = ["HanChuan", "HongHu", "LongKou", ]
    for c in dataList:
        save_path = filename + c + '\\'
        os.makedirs(save_path, exist_ok=True)
        label_all = np.load(os.path.join(filename, c, 'label.npy'))

        for tr in [1, 2, 3]:
            data = np.load(os.path.join(filename, c, str(tr), 'train_coordinates.npy'))
            print(data.shape)
            plt.imshow(label_all)
            plt.scatter(data[:, 1], data[:, 0], c='Red', s=1)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(filename, c, f'{tr}.jpg'), dpi=400, bbox_inches='tight')

            plt.close()
            # exit()
def get_pic():

    filename = 'D:\A_real_data\HSI\coordinate_新_周恒\\'
    colors2 = np.loadtxt('./color2.txt')  #
    dataList = {

        "Indian Pines": np.array([200, 145, 145, 16]),
        "salinas": np.array([204, 512, 217, 16]),
        "Pavia University": np.array([102, 610, 340, 9]),
        "HanChuan": np.array([274, 1217, 303, 17]),
        "HongHu": np.array([270, 940, 475, 22]),
        "LongKou": np.array([270, 550, 400, 10]),
    }
    for c,c_i in dataList.items():
        print(c_i,c)
        save_path = filename + c + '\\'
        os.makedirs(save_path, exist_ok=True)
        label_all = np.load(os.path.join(filename, c, 'label.npy'))
        print(c)
        for tr in [1, 2, 3]:
            data = np.load(os.path.join(filename, c, str(tr), 'train_coordinates.npy'))
            label_all2 = label_all.copy()
            plt.imshow(label_all)
            plt.scatter(data[:, 1], data[:, 0], c='Red', s=1)
            plt.xticks([])
            plt.yticks([])


            gradientImage=dilation(label_all2, square(2)) - erosion(label_all2, square(2))
            gradientImage[gradientImage!=0] = -1
            # label_all[label_all !=0 ] = 1
            label_all2[gradientImage == -1] = 100
            plt.imshow(label_all2,cmap='Greys')
            # plt.show()
            plt.savefig(os.path.join(filename, c, f'{tr}.jpg'), dpi=400, bbox_inches='tight', pad_inches = -0.1)
            plt.close()
            # pic = get_pre_pic(label_all, colors2, c_i[1:3])
            # imsave(save_path + f'{c}_{tr}_标注黑框.png', pic)
            # exit()
if __name__ == '__main__':
    # get_index_pic() # 画数据的索引坐标点到
    get_pic() # 画图像闭包