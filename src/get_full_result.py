'''
Description: Used in getting results of all types
Author:Charles Shen
Date:8/29/2020
'''
import torch
import argparse
from data.shapenet_loader import load_shapenet_all, load_shapenet_type
from models.PCN import PCN
from torch.utils.data import DataLoader
from utils.load_models import init_trained_model, init_trained_PCN
import ast
import os
import numpy as np
import constants
import chamfer3D.dist_chamfer_3D
from utils.common import get_triplet_loss, get_chamfer_dist_train, get_chamfer_dist_valid, step_weight

DATA_PATH = '/data1/xp/shapenet'
MODEL_BASE_DIR = '../visualize/models'
MODEL_DIR = 'base'


def valid_result_one(args, device, generator_partial, generator_complete, decoder, data_loader_shapenet_val):
    generator_partial.eval()
    generator_complete.eval()
    decoder.eval()
    
    total_dist = 0
    total_batch = 0
    for i, (partial_shapenet, ground_truth_fine, ground_truth_coarse)  in enumerate(data_loader_shapenet_val):
        if device:
            partial_shapenet = partial_shapenet.to(device)
            ground_truth_fine = ground_truth_fine.to(device)
            ground_truth_coarse = ground_truth_coarse.to(device)
            #partial_scannet = partial_scannet.to(device)
        
        if args.dataset == 'complete':
            batch_size = partial_shapenet.size(0)
            num_partial = partial_shapenet.size(1)
            partial_shapenet = partial_shapenet.resize(batch_size * num_partial, partial_shapenet.size(2), partial_shapenet.size(3))
            ground_truth_fine = ground_truth_fine.repeat(num_partial, 1, 1)
            ground_truth_coarse = ground_truth_coarse.repeat(num_partial, 1, 1)

        feature_anchor = generator_partial(partial_shapenet)
        feature_positive = generator_complete(partial_shapenet)
        if args.loss == 'cosine':
            #归一化
            if args.normalize == True:
                feature_anchor = torch.nn.functional.normalize(feature_anchor, dim = 1)
        if args.dataset == 'complete':
            coarse_anchor, fine_anchor = decoder(feature_anchor)
            dis_anchor = get_chamfer_dist_valid(coarse_anchor, fine_anchor, ground_truth_coarse, ground_truth_fine)

            coarse_positive, fine_positive = decoder(feature_positive)
            dis_positive = get_chamfer_dist_valid(coarse_positive, fine_positive, ground_truth_coarse, ground_truth_fine)

            total_dist += min(dis_anchor.item(), dis_positive.item()) * 10000
            total_batch += 1
        else:
            coarse_anchor, fine_anchor = decoder(feature_anchor)
            dis_anchor = get_chamfer_dist_valid(coarse_anchor, fine_anchor, ground_truth_coarse, ground_truth_fine)
            total_dist += dis_anchor.item() * 10000
            total_batch += 1
    
    avg_dist = total_dist / total_batch
    print('total average dist: {:.2f}'.format(avg_dist))


def valid_PCN(args, device, model, data_loader_shapenet_val):
    
    model.eval()
    
    total_dist = 0
    total_batch = 0
    for i, (partial_shapenet, ground_truth_fine, ground_truth_coarse)  in enumerate(data_loader_shapenet_val):
        if device:
            partial_shapenet = partial_shapenet.to(device)
            ground_truth_fine = ground_truth_fine.to(device)
            ground_truth_coarse = ground_truth_coarse.to(device)
            
        if args.dataset == 'complete':
            batch_size = partial_shapenet.size(0)
            num_partial = partial_shapenet.size(1)
            partial_shapenet = partial_shapenet.resize(batch_size * num_partial, partial_shapenet.size(2), partial_shapenet.size(3))
            ground_truth_fine = ground_truth_fine.repeat(num_partial, 1, 1)
            ground_truth_coarse = ground_truth_coarse.repeat(num_partial, 1, 1)
        
        coarse, fine = model(partial_shapenet)
        dis = get_chamfer_dist_valid(coarse, fine, ground_truth_coarse, ground_truth_fine)


        total_dist += dis.item() * 10000
        total_batch += 1

    avg_dist = total_dist / total_batch
    print('total average dist: {:.2f}'.format(avg_dist))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Get full result')


    parser.add_argument('--seed', type = int, default = 1234, help='set random seed')
    parser.add_argument('--model', type = str, default = 'mine', help = 'use which model?')

    parser.add_argument('--cuda',  type = ast.literal_eval, default = True, help = 'use GPU or not')
    parser.add_argument('--gpu_id', type = int, default = 8, help = 'GPU device id used')
    parser.add_argument('--dataset', type = str, default = 'shapenet', help = 'use which?shapenet or complete?')
    parser.add_argument('--loss', type = str, default = 'cosine', help = 'use which loss?')
    parser.add_argument('--model_dir', type = str, default = MODEL_DIR, help='model dir')
    parser.add_argument('--normalize', type = ast.literal_eval, default = False, help = 'whether normalize feature before cosine loss or not')

    args = parser.parse_args()
    print(args)

    print('getting device...', end='')
    torch.manual_seed(args.seed)
    if args.cuda == True:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(args.gpu_id)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    print('device got')


    model_path = os.path.join(MODEL_BASE_DIR, args.model_dir)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if args.model == 'mine':
        generator_partial, generator_complete, decoder = init_trained_model(MODEL_BASE_DIR, args.model_dir, None)
        if device:
            generator_partial.to(device)
            generator_complete.to(device)
            decoder.to(device)

        for the_type in constants.types.keys():
            print('getting dataset')
            dataset_shapenet_train, dataset_shapenet_val = load_shapenet_type(DATA_PATH, constants.types[the_type], 512)  
            print('getting dataloader')
            data_loader = DataLoader(dataset_shapenet_val, batch_size = constants.batch_size, shuffle=True, num_workers=2)
            print('data got!')
            print(the_type)
            valid_result_one(args, device, generator_partial, generator_complete, decoder, data_loader)
    else:
        model = init_trained_PCN(MODEL_BASE_DIR, args.model_dir, None)
        if device:
            model.to(device)


        for the_type in constants.types.keys():
            print('getting dataset')
            dataset_shapenet_train, dataset_shapenet_val = load_shapenet_type(DATA_PATH, constants.types[the_type], 512)  
            print('getting dataloader')
            data_loader = DataLoader(dataset_shapenet_val, batch_size = constants.batch_size, shuffle=True, num_workers=2)
            print('data got!')
            print(the_type)
            valid_PCN(args, device, model, data_loader)