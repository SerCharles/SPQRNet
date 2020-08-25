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
        gt_list
        partial_list
        num_coarse: num of coarse pointcloud, used in deriving coarse gt
    '''
    def __init__(self, gt_list, partial_list, num_coarse = 512):
        self.gt = torch.stack(gt_list)
        self.partial = torch.stack(partial_list, 0)
        self.gt_coarse = index_points(self.gt, farthest_point_sample(self.gt, num_coarse))

    def __len__(self):
        return self.gt.shape[0]

    def __getitem__(self, idx):
        return self.partial[idx], self.gt[idx], self.gt_coarse[idx]

def load_shapenet(base_dir, type_code):
    '''
    Load one type of data
    Variables：
        base_dir: the file dir of shapenet
        type_code: typecode of the type
    Return: gt_list_train, partial_list_train, gt_list_valid, partial_list_valid
    '''
    dir_gt_train = os.path.join(base_dir, 'train', 'gt', type_code)
    dir_partial_train = os.path.join(base_dir, 'train', 'partial', type_code)
    dir_gt_valid = os.path.join(base_dir, 'val', 'gt', type_code)
    dir_partial_valid = os.path.join(base_dir, 'val', 'partial', type_code)

    model_names = os.listdir(dir_gt_train)
    gt_list_train = []
    partial_list_train = []
    for filename in model_names:
        try:
            the_gt = torch.from_numpy(h5py.File(os.path.join(dir_gt_train, filename), 'r')['data'][:]).float()
            the_partial = torch.from_numpy(h5py.File(os.path.join(dir_partial_train, filename), 'r')['data'][:]).float()
            gt_list_train.append(the_gt)
            partial_list_train.append(the_partial)
        except:
            print("Loading data " + filename + " failed")

    model_names = os.listdir(dir_gt_valid)
    gt_list_valid = []
    partial_list_valid = []
    for filename in model_names:
        try:
            the_gt = torch.from_numpy(h5py.File(os.path.join(dir_gt_valid, filename), 'r')['data'][:]).float()
            the_partial = torch.from_numpy(h5py.File(os.path.join(dir_partial_valid, filename), 'r')['data'][:]).float()
            gt_list_valid.append(the_gt)
            partial_list_valid.append(the_partial)
        except:
            print("Loading data " + filename + " failed")
    return gt_list_train, partial_list_train, gt_list_valid, partial_list_valid

def load_shapenet_type(base_dir, type_code, num_coarse = 512):
    '''
    Load one specific type of data, interface
    Variables：
        base_dir: the file dir of shapenet
        type_code: typecode of the type
        num_coarse: num of coarse pointcloud, used in deriving coarse gt
    Return:
        dataset_train, dataset_valid
    '''
    gt_list_train, partial_list_train, gt_list_valid, partial_list_valid = load_shapenet(base_dir, type_code)
    dataset_train = ShapeNetLoader(gt_list_train, partial_list_train, num_coarse)
    dataset_valid = ShapeNetLoader(gt_list_valid, partial_list_valid, num_coarse)
    return dataset_train, dataset_valid

def load_shapenet_all(base_dir, types, num_coarse = 512):
    '''
    Load all types of data, interface
    Variables：
        base_dir: the file dir of shapenet
        types: typecodes of the type
        num_coarse: num of coarse pointcloud, used in deriving coarse gt
    Return:
        dataset_train, dataset_valid
    '''
    gt_list_train = []
    partial_list_train = []
    gt_list_valid = []
    partial_list_valid = []
    for (type_name, type_code) in types.items():
        the_gt_list_train, the_partial_list_train, the_gt_list_valid, the_partial_list_valid = load_shapenet(base_dir, type_code)
        gt_list_train += the_gt_list_train
        gt_list_valid += the_gt_list_valid
        partial_list_train += the_partial_list_train
        partial_list_valid += the_partial_list_valid
        print("Loading", str(type_name), "finished!")


    dataset_train = ShapeNetLoader(gt_list_train, partial_list_train, num_coarse)
    dataset_valid = ShapeNetLoader(gt_list_valid, partial_list_valid, num_coarse)    
    return dataset_train, dataset_valid