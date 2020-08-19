'''
Description:The feature projectors of the network, DNN
Author:Charles Shen
Data:8/17/2020
'''


import torch 
import torch.nn as nn
import torchvision



class Projector(nn.Module):
    '''
    参数:
        in_size：输入的数据维数，比如参数1024，则输入b*1024维
        out_size:输出的数据维数，比如参数256，则输出b*256维
    输入：b*in_size的点云特征
    输出：b*out_size的投影后的特征
    '''
    def __init__(self, in_size = 1024, out_size = 256):
        super(Projector, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #shared mlp0:1d conv
        self.mlp = nn.Sequential(
            nn.Linear(in_features = self.in_size, out_features = 1024, bias = True),
            nn.LeakyReLU(),
            nn.Linear(in_features = 1024, out_features = 512, bias = True),
            nn.LeakyReLU(),
            nn.Linear(in_features = 512, out_features = out_size, bias = True),
            nn.Tanh()
        )
        

    def forward(self, x):
        return self.mlp(x)
