'''
Description:The encoder of the both sides of the Network, PCN Encoder
Author:Charles Shen
Date:8/17/2020
'''

import torch 
import torch.nn as nn
import torchvision



class Encoder(nn.Module):
    '''
        variables:
            out_size: the dimension of the output feature, 1024 means the output is b*1024
        input: partial pointcloud, which is b*n*3
        output: feature, which is b*out_size
    '''
    def __init__(self, out_size = 1024):
        super(Encoder, self).__init__()
        self.out_size = out_size

        #shared mlp0:1d conv
        self.shared_mlp0 = nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels = 128, kernel_size = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(in_channels =128, out_channels = 256, kernel_size = 1)
        )
        
        #shared mlp1:1d conv
        self.shared_mlp1 = nn.Sequential(
            nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 1),
            nn.LeakyReLU(inplace = True),
            nn.Conv1d(in_channels = 512, out_channels = self.out_size, kernel_size = 1)
        )


    def forward(self, x):
        x = x.permute(0, 2, 1) #b*3*n
        feature_f = self.shared_mlp0(x) #b*256*n
        global_feature_g, _ = torch.max(feature_f, -1, keepdim = True) #b*256*1
        expanded_global_feature_g = global_feature_g.repeat(1, 1, feature_f.size(2)) #b*256*n
        concated_feature = torch.cat([expanded_global_feature_g, feature_f], 1) #b*512*n
        global_feature_v = self.shared_mlp1(concated_feature) #b*out_size*n
        global_feature_v, _ = torch.max(global_feature_v, -1, keepdim = True) #b*out_size*1
        del _
        global_feature_v = global_feature_v.view(global_feature_v.size(0), -1) #b*out_size
        return global_feature_v