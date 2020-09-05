'''
Description: Original PCN Network
Author:Charles Shen
Date:8/17/2020
'''


import torch 
import torch.nn as nn
import torchvision
from models.encoder import Encoder
from models.decoder import Decoder
from models.fused_decoder import Decoder_fused


class PCN(nn.Module):
    '''
        variables:
            feature_size：the medium dimension of the data, 1024 means the output is b*1024
            grid_scale, grid_size: grid info used in PCN folding
            num_coarse: num of points of the coarse output pointcloud; num of the fine pointcloud is num_coarse * grid_size * grid_size
        input: the partial pointcloud of b*n*3
        output: coarse output(b * num_coarse * 3), fine output(b*(num_coarse*grid_size^2)*3)
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

class PCN_fused(nn.Module):
    '''
        variables:
            feature_size：the medium dimension of the data, 1024 means the output is b*1024
            grid_scale, grid_size: grid info used in PCN folding
            num_coarse: num of points of the coarse output pointcloud; num of the fine pointcloud is num_coarse * grid_size * grid_size
        input: the partial pointcloud of b*n*3
        output: coarse output(b * num_coarse * 3), fine output(b*(num_coarse*grid_size^2)*3)
    '''

    def __init__(self, feature_size = 1024, grid_scale = 0.05, grid_size = 2, num_coarse = 512):
        super(PCN_fused, self).__init__()
        self.encoder_partial = Encoder(out_size = feature_size)
        self.encoder_complete = Encoder(out_size = feature_size)

        self.decoder = Decoder_fused(in_size = feature_size, grid_scale = grid_scale, grid_size = grid_size, num_coarse = num_coarse)

    def forward(self, partial, complete):
        partial_feature = self.encoder_partial(partial)
        complete_feature = self.encoder_complete(complete)
        return self.decoder(partial_feature, complete_feature)

    def to(self, device, **kwargs):
        self.decoder.grid_x = self.decoder.grid_x.to(device)
        self.decoder.grid_y = self.decoder.grid_y.to(device)
        super(PCN_fused, self).to(device, **kwargs)