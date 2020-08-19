'''
Description:The discrimiator of the network, MLP
Author:Charles Shen
Data:8/17/2020
'''


import torch 
import torch.nn as nn
import torchvision


class Discriminator(nn.Module):
    '''
    参数:
        in_size：输入的数据维数，比如参数1024，则输入b*1024维
    输入：b*in_size的点云特征
    输出：b*2 分类结果
    '''
    def __init__(self, in_size = 256):
        super(Discriminator, self).__init__()
        self.in_size = in_size

        #shared mlp0:1d conv
        self.mlp = nn.Sequential(
            nn.Linear(in_features = self.in_size, out_features = 128, bias = True),
            nn.Sigmoid(),
            nn.Linear(in_features = 128, out_features = 64, bias = True),
            nn.Sigmoid(),
            nn.Linear(in_features = 64, out_features = 1, bias = True),
            nn.Sigmoid(),
        )
        

    def forward(self, x):
        return self.mlp(x)