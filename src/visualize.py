'''
Description: Used in visualizing results, only can run on my windows
Author:Charles Shen
Date:8/29/2020
'''
import torch
import argparse
from data.shapenet_loader import load_shapenet_all, load_shapenet_type
from models.PCN import PCN
from torch.utils.data import DataLoader
from utils.load_models import init_trained_model, init_trained_PCN
import os
import open3d
import numpy as np
import constants

DATA_PATH = 'E:\\dataset\\shapenet'
MODEL_BASE_DIR = '..\\visualize\\models'
RESULT_BASE_DIR = '..\\visualize\\results'
MODEL_DIR = 'base'
RESULT_DIR = 'base'
BATCH_SIZE = 1



def init_local_data(args):
    '''
        description: load data
        variable: args, base_dir, model_dir
        return: data_loader
    '''
    print('getting dataset')
    if args.all == True:
        dataset_shapenet_train, dataset_shapenet_val = load_shapenet_all(DATA_PATH, constants.types, 512)
    else:
        dataset_shapenet_train, dataset_shapenet_val = load_shapenet_type(DATA_PATH, constants.types[args.type], 512)  
    print('getting dataloader')
    data_loader = DataLoader(dataset_shapenet_val, batch_size = BATCH_SIZE, shuffle=True, num_workers=2)
    print('data got!')
    return data_loader


def visualize(args, fine, i):
    '''
        description: visualize one pointcloud
        variable: args, fine_pointcloud, i
        return: empty
    '''
    fine_np = fine[0].detach().numpy()
    fine_pcd = open3d.geometry.PointCloud()
    fine_pcd.points = open3d.utility.Vector3dVector(fine_np)
    result_name = 'result_' + str(i) + '.ply'
    open3d.io.write_point_cloud(os.path.join(RESULT_BASE_DIR, args.result_dir, result_name), fine_pcd)
    print('written ', result_name)


def visualize_result(args, data_loader, encoder_anchor, encoder_positive, decoder):
    '''
        description: visualize result of my own model
        variable: args, data_loader, encoder_anchor, encoder_positive, decoder
        return: empty
    '''

    for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):
        feature_anchor = encoder_anchor(the_data)
        coarse_anchor, fine_anchor = decoder(feature_anchor)
        visualize(args, fine_anchor, i)

def visualize_baseline(args, data_loader, model):
    '''
        description: visualize result of my PCN
        variable: args, data_loader, model
        return: empty
    '''
    for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):
        coarse, fine = model(the_data)
        visualize(args, fine, i)


def visualize_gt(args, data_loader):
    '''
        description: visualize gts
        variable: args, data_loader
        return: empty
    '''
    for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):
        visualize(args, ground_truth_fine, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Visualize')
    parser.add_argument('--use_model', type = str, default = 'main', help='use model, empty:gt PCN:PCN main:my model')
    parser.add_argument('--all', type = bool, default = False, help='visualize all data or not')
    parser.add_argument('--type', type = str, default = 'chair', help='show which type')
    parser.add_argument('--model_dir', type = str, default = MODEL_DIR, help='model dir')
    parser.add_argument('--result_dir', type = str, default = RESULT_DIR, help='result dir')
    args = parser.parse_args()
    print(args)
    
    model_path = os.path.join(MODEL_BASE_DIR, args.model_dir)
    result_path = os.path.join(RESULT_BASE_DIR, args.result_dir)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    data_loader = init_local_data(args)

    if args.use_model == 'main':
        encoder_anchor, encoder_positive, decoder = init_trained_model(MODEL_BASE_DIR, args.model_dir, None)
        visualize_result(args, data_loader, encoder_anchor, encoder_positive, decoder)
    elif args.use_model == 'PCN':
        model = init_trained_PCN(MODEL_BASE_DIR, args.model_dir, None)
        visualize_baseline(args, data_loader, model)
    else:
        visualize_gt(args, data_loader)