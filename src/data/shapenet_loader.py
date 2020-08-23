'''
Description:The data loader of shapenet
Author:Charles Shen
Data:8/18/2020
'''

import os
import glob
import numpy as np
import torch
import h5py
from utils.pointnet_util import farthest_point_sample, index_points
from torch.utils.data import Dataset


class ShapeNetLoader(Dataset):
    '''
    Variables：
        base_dir: the file dir of shapenet
        data_type: string indicating train or val
        type_code: typecode of the type
        num_coarse: num of coarse pointcloud, used in deriving coarse gt
    '''
    def __init__(self, base_dir, data_type, type_code, num_coarse = 512):
        dir_gt = os.path.join(base_dir, data_type, 'gt', type_code)
        dir_partial = os.path.join(base_dir, data_type, 'partial', type_code)

        model_names = os.listdir(dir_gt)
        gt_list = []
        partial_list = []
        for filename in model_names:
            try:
                the_gt = torch.from_numpy(h5py.File(os.path.join(dir_gt, filename), 'r')['data'][:]).float()
                the_partial = torch.from_numpy(h5py.File(os.path.join(dir_partial, filename), 'r')['data'][:]).float()
                gt_list.append(the_gt)
                partial_list.append(the_partial)
            except:
                print("Loading data " + filename + " failed")

        self.gt = torch.stack(gt_list)
        self.partial = torch.stack(partial_list, 0)
        self.gt_coarse = index_points(self.gt, farthest_point_sample(self.gt, num_coarse))

    def __len__(self):
        return self.gt.shape[0]

    def __getitem__(self, idx):
        return self.partial[idx], self.gt[idx], self.gt_coarse[idx]