import os
import sys
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from . import tools


class Feeder(torch.utils.data.Dataset):
    """
    Feeder用于基于骨骼的动作识别数据输入和改变
    Arguments:
        data_path:".npy"数据文件路径，shape为 N,C,F,J,M
        label_path:".pkl"标签文件路径
        random_choose:随机选择输入序列的一部分
        random_move：对数据进行变换
        window_size:输出的数据长度
        debug:用于测试的sample数据，前100个sample数据
        mmap:以制定格式将大型数据文件映射到内存中
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=True,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.debug = debug

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C F J M
        # 加载标签label
        with open(self.label_path, "rb") as f:
            self.sample_name, self.label = pickle.load(f)

        # 加载数据data
        # 对文件进行内存映射，以处理超大容量文件
        if mmap:
            self.data = np.load(self.data_path, mmap_mode="r")
        else:
            self.data = np.load(self.data_path)

        # 检测debug测试,选取前100个文件
        if self.debug:
            self.sample_name = self.sample_name[0:50]
            self.data = self.data[0:50]
            self.label = self.label[0:50]

        self.N, self.C, self.F, self.J, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # 得到数据
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy,label


class MultiDataset(torch.utils.data.Dataset):
    """
    将多个视角下的数据集对象进行整合处理，方便后续同步打乱
    """
    def __init__(self, view1, view2, view3):
        self.view1 = view1
        self.view2 = view2
        self.view3 = view3

    def __len__(self):
        return len(self.view1)

    def __getitem__(self, index):
        index1 = self.view1[index]
        index2 = self.view2[index]
        index3 = self.view3[index]
        return index1, index2, index3













