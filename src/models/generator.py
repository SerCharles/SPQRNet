'''
Description:The generator of the network
Author:Charles Shen
Data:8/17/2020
'''


import torch 
import torch.nn as nn
import torchvision
from models.encoder import Encoder
from models.projector import Projector

class Generator(nn.Module):
    '''
    Variables:
        feature_size:提取点云特征维数，如果是1024就b*1024
        project_size:投影特征维数，如果是256就b*256
    Input：b*n1*3点云
    Output：点云特征b*project_size
    '''
    def __init__(self, feature_size = 1024, project_size = 256):
        super(Generator, self).__init__()
        self.feature_size = feature_size
        self.project_size = project_size

        self.encoder = Encoder(out_size = self.feature_size)

        self.projector = Projector(in_size = self.feature_size, out_size = self.project_size)

        

    def forward(self, x):
        feature = self.projector(self.encoder(x))
        return feature


