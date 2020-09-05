'''
Description:The decoder of the Network, PCN Decoder
Author:Charles Shen
Date:8/17/2020
'''

import torch 
import torch.nn as nn
import torchvision

class Decoder_fused(nn.Module):
    '''
        variables:
            in_size: the input dimension of the medium feature, 1024 means the input is b*1024
            grid_scale, grid_size: grid info used in PCN folding
            num_coarse: num of points of the coarse output pointcloud; num of the fine pointcloud is num_coarse * grid_size * grid_size
        input: medium feature derived by the encoder, b*in_size
        output: coarse output(b * num_coarse * 3), fine output(b*(num_coarse*grid_size^2)*3)
    '''
    def __init__(self, in_size = 1024, grid_scale = 0.05, grid_size = 2, num_coarse = 512):
        super(Decoder_fused, self).__init__()
        self.in_size = in_size
        self.grid_scale = grid_scale
        self.grid_size = grid_size
        self.num_coarse = num_coarse
        self.num_fine = num_coarse * grid_size * grid_size

        #mlp0
        self.mlp1 = nn.Sequential(
            nn.Linear(in_features = self.in_size, out_features = 1024, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 1024, out_features = 1024, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 1024, out_features = num_coarse * 3 , bias = True)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(in_features = self.in_size, out_features = 1024, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 1024, out_features = 1024, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 1024, out_features = num_coarse * 3 , bias = True)
        )
        
        #folding_mlp:1d conv
        self.folding_mlp = nn.Sequential(
            nn.Conv1d(in_channels = 1024 * 2 + 3 * 2 + 2, out_channels = 512, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels = 512, out_channels = 3, kernel_size = 1)
        )

        #grid
        x = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        y = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        self.grid_x, self.grid_y = torch.meshgrid(x, y)

    def forward(self, feature_partial, feature_complete):
        coarse1 = self.mlp1(feature_partial)  #b * (3 * coarse_size)
        coarse2 = self.mlp2(feature_complete)
        coarse1 = coarse1.view(coarse1.size(0), 3, -1) #B * 3 * coarse_size
        coarse2 = coarse2.view(coarse2.size(0), 3, -1) 
        coarse = torch.div((coarse1 + coarse2), 2.00)
        #print(coarse.size())

        center = coarse.view(coarse.size(0), coarse.size(1), coarse.size(2), 1) #B * 3 * coarse_size * 1
        center = center.repeat(1, 1, 1, self.grid_size ** 2) #B * 3 * coarse_size, t
        center = center.view(center.size(0), center.size(1), -1) #B * 3 * fine_size

        coarse1 = coarse1.view(coarse1.size(0), coarse1.size(1), coarse1.size(2), 1)
        coarse1 = coarse1.repeat(1, 1, 1, self.grid_size ** 2) #B * 3 * coarse_size, t
        coarse1 = coarse1.view(coarse1.size(0), coarse1.size(1), -1) #B * 3 * fine_size

        coarse2 = coarse2.view(coarse2.size(0), coarse2.size(1), coarse2.size(2), 1)
        coarse2 = coarse2.repeat(1, 1, 1, self.grid_size ** 2) #B * 3 * coarse_size, t
        coarse2 = coarse2.view(coarse2.size(0), coarse2.size(1), -1) #B * 3 * fine_size
        #print(center.size())
        
        #生成folding用的x，y网格参数
        grid = torch.cat([self.grid_x, self.grid_y], -1) #u*u*2
        grid = grid.view(1, 2, self.grid_size ** 2) #1 * 2 * t
        expanded_grid = grid.repeat(1, 1, self.num_coarse) #1 * 2 * (t*coarse_size) = 1*2*(fine_size)
        expanded_grid = expanded_grid.repeat(feature_partial.size(0), 1, 1) #B * 2* (fine_size)
        #print(expanded_grid.size())

        #生成encoder信息对应的参数
        encoder_info_partial = feature_partial.view(feature_partial.size(0), feature_partial.size(1), 1) #B * 1024 * 1
        encoder_info_partial = encoder_info_partial.repeat(1, 1, self.num_fine)#B*1024*fine_size
        encoder_info_complete = feature_complete.view(feature_complete.size(0), feature_complete.size(1), 1) #B * 1024 * 1
        encoder_info_complete = encoder_info_complete.repeat(1, 1, self.num_fine)#B*1024*fine_size
        #print(encoder_info.size())


        
        #结合得到整个feature
        full_feature = torch.cat([expanded_grid, coarse1, coarse2, encoder_info_partial, encoder_info_complete], 1) #B * 2056 * fine_size
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
        super(Decoder_fused, self).to(device, **kwargs)