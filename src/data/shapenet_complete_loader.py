'''
Description:The data loader of shapenet completion
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


class ShapeNetCompleteLoader(Dataset):
    '''
    Variables：
        num_coarse: num of coarse pointcloud, used in deriving coarse gt
        gt_list: the list of ground truth
        partial_list: the list of partial
    '''
    def __init__(self, gt_list, partial_list, num_coarse = 512):
        self.gt = torch.stack(gt_list)
        self.partial = torch.stack(partial_list, 0)
        self.gt_coarse = index_points(self.gt, farthest_point_sample(self.gt, num_coarse))

    def __len__(self):
        return self.gt.shape[0]

    def __getitem__(self, idx):
        return self.partial[idx], self.gt[idx], self.gt_coarse[idx]


def load_shapenet(base_dir, type_name, type_code, num_partial = 8):
    '''
    Description: load shapenet data of one type
    Variables：
        base_dir: the file dir of shapenet
        type_name: type name of the type
        type_code: type code of the type
        num_partial: the number of partial pointclouds corresponding to the fine one
    Return:
        gt_list_train, partial_list_train, gt_list_valid, partial_list_valid
    '''
    dir_gt = os.path.join(base_dir, type_name, 'gt')
    dir_partial = os.path.join(base_dir, type_name, 'train')
    dir_list = os.path.join(base_dir, 'test_models.txt')
    test_file = open(dir_list, 'r')
    test_list_raw = test_file.readlines()
    test_file.close()
    test_list = []
    for item in test_list_raw:
        split_item = item.strip().split('\\')
        the_code = split_item[0]
        the_name = split_item[1]
        if the_code == type_code:
            test_list.append(the_name)
    
    model_names = os.listdir(dir_gt)
    gt_list_train = []
    partial_list_train = []
    gt_list_valid = []
    partial_list_valid = []
    for filename in model_names:
        try:
            file_id = filename.split('.')[0]
            the_gt = torch.from_numpy(np.load(os.path.join(dir_gt, filename))).float()
            current_partial_list = []
            for i in range(num_partial):
                partial_name = file_id + '__' + str(i) + '__' + '.ply.npy'
                current_partial = torch.from_numpy(np.load(os.path.join(dir_partial, partial_name))).float()
                current_partial_list.append(current_partial)
            the_partial = torch.stack(current_partial_list, 0)
            #print(the_partial.size())

            if file_id in test_list:
                gt_list_valid.append(the_gt)
                partial_list_valid.append(the_partial)
            else:
                gt_list_train.append(the_gt)
                partial_list_train.append(the_partial)
        except:
            print("Loading data " + filename + " failed")
    return gt_list_train, partial_list_train, gt_list_valid, partial_list_valid


def init_shapenet_complete_data(base_dir, type_name, type_code, num_partial = 8):
    '''
    Description: load shapenet data of one type, interface
    Variables：
        base_dir: the file dir of shapenet
        type_name: type name of the type
        type_code: type code of the type
        num_partial: the number of partial pointclouds corresponding to the fine one
    Return:
        train_dataset, valid_dataset, of one type
    '''
    gt_list_train, partial_list_train, gt_list_valid, partial_list_valid = load_shapenet(base_dir, type_name, type_code, 8)
    train_dataset = ShapeNetCompleteLoader(gt_list_train, partial_list_train)
    valid_dataset = ShapeNetCompleteLoader(gt_list_valid, partial_list_valid)
    return train_dataset, valid_dataset


def init_shapenet_complete_data_all(base_dir, types, num_partial = 8):
    '''
    Description: load shapenet data of all types, interface
    Variables：
        base_dir: the file dir of shapenet
        type_name: type name of the type
        type_code: type code of the type
        num_partial: the number of partial pointclouds corresponding to the fine one
    Return:
        train_dataset, valid_dataset, of all types
    '''
    gt_list_train = []
    partial_list_train = []
    gt_list_valid = []
    partial_list_valid = []
    print(types)
    for (type_name, type_code) in types.items():
        the_gt_list_train, the_partial_list_train, the_gt_list_valid, the_partial_list_valid = load_shapenet(base_dir, type_name, type_code, 8)
        gt_list_train += the_gt_list_train
        partial_list_train += the_partial_list_train
        gt_list_valid += the_gt_list_valid
        partial_list_valid += the_partial_list_valid

    train_dataset = ShapeNetCompleteLoader(gt_list_train, partial_list_train)
    valid_dataset = ShapeNetCompleteLoader(gt_list_valid, partial_list_valid)
    return train_dataset, valid_dataset
