'''
Description:The decoder of the Network, PCN Decoder
Author:Charles Shen
Date:8/17/2020
'''

import torch 
import torch.nn as nn
import torchvision

class Decoder(nn.Module):
    '''
        variables:
            in_size: the input dimension of the medium feature, 1024 means the input is b*1024
            grid_scale, grid_size: grid info used in PCN folding
            num_coarse: num of points of the coarse output pointcloud; num of the fine pointcloud is num_coarse * grid_size * grid_size
        input: medium feature derived by the encoder, b*in_size
        output: coarse output(b * num_coarse * 3), fine output(b*(num_coarse*grid_size^2)*3)
    '''
    def __init__(self, in_size = 1024, grid_scale = 0.05, grid_size = 2, num_coarse = 512):
        super(Decoder, self).__init__()
        self.in_size = in_size
        self.grid_scale = grid_scale
        self.grid_size = grid_size
        self.num_coarse = num_coarse
        self.num_fine = num_coarse * grid_size * grid_size

        #mlp0
        self.mlp = nn.Sequential(
            nn.Linear(in_features = self.in_size, out_features = 1024, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Linear(in_features = 1024, out_features = 1024, bias = True),
            nn.LeakyReLU(inplace = True),
            nn.Linear(in_features = 1024, out_features = num_coarse * 3 , bias = True)
        )
        
        #folding_mlp:1d conv
        self.folding_mlp = nn.Sequential(
            nn.Conv1d(in_channels = 1024 + 3 + 2, out_channels = 512, kernel_size = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(in_channels = 512, out_channels = 3, kernel_size = 1)
        )

        #grid
        x = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        y = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        self.grid_x, self.grid_y = torch.meshgrid(x, y)

    def forward(self, feature):
        #生成coarse的参数
        coarse = self.mlp(feature) #b * (3 * coarse_size)
        coarse = coarse.view(coarse.size(0), 3, -1) #B * 3 * coarse_size
        #print(coarse.size())
        center = coarse.view(coarse.size(0), coarse.size(1), coarse.size(2), 1)
        center = center.repeat(1, 1, 1, self.grid_size ** 2)
        center = center.view(center.size(0), center.size(1), -1)
        #print(center.size())
        
        #生成folding用的x，y网格参数
        grid = torch.cat([self.grid_x, self.grid_y], -1) #u*u*2
        grid = grid.view(1, 2, self.grid_size ** 2) #1 * 2 * t
        expanded_grid = grid.repeat(1, 1, self.num_coarse) #1 * 2 * (t*coarse_size) = 1*2*fine_size
        expanded_grid = expanded_grid.repeat(feature.size(0), 1, 1) #B * 2* fine_size
        #print(expanded_grid.size())

        #生成encoder信息对应的参数
        encoder_info = feature.view(feature.size(0), feature.size(1), 1) #B*1024*1
        encoder_info = encoder_info.repeat(1, 1, self.num_fine)#B*1024*fine_size
        #print(encoder_info.size())

        #结合得到整个feture
        full_feature = torch.cat([expanded_grid, center, encoder_info], 1) #B * 1029 *fine_size
        fine = self.folding_mlp(full_feature) #B*3*fine_size
        fine = fine + center #B*3*fine_size
        coarse = coarse.permute(0, 2, 1) #B*coarse_size*3
        fine = fine.permute(0, 2, 1) #B*fine_size * 3
        #print(coarse.size())
        #print(fine.size())
        return coarse, fine

    def to(self, device, **kwargs):
        self.grid_x = self.grid_x.to(device)
        self.grid_y = self.grid_y.to(device)
        super(Decoder, self).to(device, **kwargs)