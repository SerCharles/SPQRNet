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
    参数:
        feature_size:提取点云特征维数，如果是1024就b*1024
        project_size:投影特征维数，如果是256就b*256
    输入：有标b*n1*3点云特征，无标b*n2*3点云特征
    输出：有标投影特征，无标投影特征，都是b*project_size
    '''
    def __init__(self, feature_size = 1024, project_size = 256):
        super(Encoder, self).__init__()
        self.feature_size = feature_size
        self.project_size = project_size

        self.encoder_labeled = Encoder(out_size = self.feature_size)
        self.encoder_unlabeled = Encoder(out_size = self.feature_size)

        self.projector_labeled = Projector(in_size = self.feature_size, out_size = self.project_size)
        self.projector_unlabeled = Projector(in_size = self.feature_size, out_size = self.project_size)

        

    def forward(self, labeled_x, unlabeled_x):
        feature_labeled = self.projector_labeled(self.encoder_labeled(labeled_x))
        feature_unlabeled = self.projector_unlabeled(self.encoder_unlabeled(unlabeled_x))
        return feature_labeled, feature_unlabeled
