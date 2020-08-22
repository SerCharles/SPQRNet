'''
Description:The generator of the network
Author:Charles Shen
Data:8/17/2020
'''


import torch 
import torch.nn as nn
import torchvision
from models.encoder import Encoder
from models.decoder import Decoder


class PCN(nn.Module):
    '''
    参数:
        feature_size：中间的特征维数，比如参数1024，则中间特征b*1024维
        grid_scale, grid_size: grid大小
        num_coarse:粗糙点云的个数，默认512.精细点云个数是num_coarse * grid_size * grid_size
    输入：b*n*3的点云
    输出：b*out_size的特征
    '''
    def __init__(self, feature_size = 1024, grid_scale = 0.05, grid_size = 2, num_coarse = 512):
        super(PCN, self).__init__()
        self.encoder = Encoder(out_size = feature_size)
        self.decoder = Decoder(in_size = feature_size, grid_scale = grid_scale, grid_size = grid_size, num_coarse = num_coarse)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def to(self, device, **kwargs):
        self.decoder.grid_x = self.decoder.grid_x.to(device)
        self.decoder.grid_y = self.decoder.grid_y.to(device)
        super(PCN, self).to(device, **kwargs)

