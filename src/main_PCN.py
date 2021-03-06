'''
Description: Main program used in train and testing PCN
Author:Charles Shen
Date:8/20/2020
'''

import numpy as np
import time
import torch
import os
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.encoder import Encoder
from models.decoder import Decoder
from models.PCN import PCN
from data.scannet_loader import ScanNetLoader
from data.shapenet_loader import ShapeNetLoader
from utils.triplet_loss import random_sample, random_sample_original
from torch.utils.data import DataLoader
import chamfer3D.dist_chamfer_3D
import constants
from initialize import initialize, build_args
from utils.common import get_chamfer_dist_train, get_chamfer_dist_valid


def train(args, epoch, epochs, device, model, optimizer_PCN,data_loader_shapenet_train, result_dir_PCN):
    ''' 
        description: train the PCN model for one epoch
        variable: args, epoch, epochs, device, model, optimizer_PCN, data_loader_shapenet_train, result_dir_PCN
        return: model
    '''
    model.train()
    total_dist = 0
    total_batch = 0
    for i, (partial_shapenet, ground_truth_fine, ground_truth_coarse)  in enumerate(data_loader_shapenet_train):
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
        

        #reconstruction loss
        coarse, fine = model(partial_shapenet)
        dis = get_chamfer_dist_train(coarse, fine, ground_truth_coarse, ground_truth_fine)
            
        total_dist += dis.item() * 10000
        total_batch += 1

        optimizer_PCN.zero_grad()
        dis.backward()
        optimizer_PCN.step()

        print('Train:epoch:[{}/{}] batch {}, dis: {:.2f}'.format(epoch + 1, epochs, i+1, dis.item() * 10000))

    avg_dist = total_dist / total_batch
    file = open(result_dir_PCN, "a")
    file.write(str(avg_dist) + '\n')
    file.close()

    return model

def valid(args, epoch, epochs, device, model, data_loader_shapenet_val, best_dist, result_dir_PCN, model_dir_PCN):
    '''
        description: valid the models for one epoch
        variable: args, epoch, epochs, device, model, data_loader_shapenet_val, best_dist, result_dir_PCN, model_dir_PCN
        return: best_dist
    '''
    
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
        print('Valid:epoch:[{}/{}] batch {}, dis: {:.2f}'.format(epoch + 1, epochs, i+1, dis.item() * 10000))   

    avg_dist = total_dist / total_batch
    print('Valid:epoch:[{}/{}] total average dist: {:.2f}'.format(epoch + 1, epochs, avg_dist))
    file = open(result_dir_PCN, "a")
    file.write(str(avg_dist) + '\n')
    file.close()
    if avg_dist < best_dist:
        best_dist = avg_dist
        torch.save(model.state_dict(), model_dir_PCN)
    return best_dist



if __name__ == "__main__":
    args = build_args()
    device, generator_partial, generator_complete, decoder, pcn, decoder_fused, optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, optimizer_PCN, optimizer_fused, \
           data_loader_shapenet_train, data_loader_shapenet_val, result_dir_PCN, result_dir, model_dir_PCN, model_dir_partial, model_dir_complete, model_dir_decoder = initialize(args)
    try:
        os.remove(result_dir_PCN)
    except:
        pass


    for epoch in range(args.epochs):
        PCN = train(args, epoch, args.epochs, device, pcn, optimizer_PCN, data_loader_shapenet_train, result_dir_PCN)
        best_dist = valid(args, epoch, args.epochs, device, pcn, data_loader_shapenet_val, 14530529, result_dir_PCN, model_dir_PCN)
