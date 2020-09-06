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
from models.PCN import PCN_fused
from data.scannet_loader import ScanNetLoader
from data.shapenet_loader import ShapeNetLoader
from utils.triplet_loss import random_sample, random_sample_original, random_sample_with_self
from torch.utils.data import DataLoader
import chamfer3D.dist_chamfer_3D
import constants
from initialize import initialize, build_args
from utils.common import get_triplet_loss, get_chamfer_dist_train, get_chamfer_dist_valid, step_weight



        

def train(args, epoch, epochs, device, generator_partial, generator_complete, decoder_fused,\
    optimizer_generator_complete, optimizer_generator_partial, optimizer_fused, data_loader_shapenet_train, result_dir):
    ''' 
        description: train the PCN model for one epoch
        variable: args, epoch, epochs, device, model, optimizer_fused, data_loader_shapenet_train, result_dir_PCN
        return: empty
    '''
    generator_partial.eval()
    generator_complete.eval()
    decoder_fused.eval()
    total_dist = 0
    total_triplet = 0
    total_batch = 0
    for i, (partial_shapenet, ground_truth_fine, ground_truth_coarse)  in enumerate(data_loader_shapenet_train):
        if device:
            partial_shapenet = partial_shapenet.to(device)
            ground_truth_fine = ground_truth_fine.to(device)
            ground_truth_coarse = ground_truth_coarse.to(device)
            #partial_scannet = partial_scannet.to(device)

        if args.sample == True:
            anchor_examples, positive_examples, negative_examples = random_sample_with_self(partial_shapenet, ground_truth_fine)
        else:
            anchor_examples = partial_shapenet
            positive_examples = ground_truth_fine
            negative_examples = random_sample_original(partial_shapenet, ground_truth_fine)
        feature_anchor = generator_partial(anchor_examples)
        feature_positive = generator_complete(positive_examples)
        feature_negative = generator_complete(negative_examples)


            
    
        triplet_loss, the_times_triplet, feature_anchor, feature_positive, feature_negative = \
            get_triplet_loss(feature_anchor, feature_positive, feature_negative, args, device)


        #reconstruction loss
        #anchor
        weight_triplet = step_weight(args, epoch, epochs, the_times_triplet)

        coarse_anchor, fine_anchor = decoder_fused(feature_anchor, feature_positive)
        dis_anchor = get_chamfer_dist_train(coarse_anchor, fine_anchor, ground_truth_fine, ground_truth_coarse)
        total_loss = triplet_loss * (weight_triplet / 10000) + dis_anchor


        total_dist += dis_anchor.item() * 10000
        total_triplet += triplet_loss.item()
        total_batch += 1


        optimizer_generator_complete.zero_grad()
        optimizer_generator_partial.zero_grad()
        optimizer_fused.zero_grad()
        total_loss.backward()
        optimizer_generator_complete.step()
        optimizer_generator_partial.step()
        optimizer_fused.step()

        min_loss = dis_anchor.item()
        print('Train:epoch:[{}/{}] batch {}, dis: {:.2f}, triplet: {:.6f}'.format(epoch + 1, epochs, i+1, min_loss * 10000, triplet_loss.item()))

    avg_dist = total_dist / total_batch
    avg_triplet = total_triplet / total_batch
    file = open(result_dir, "a")
    file.write(str(avg_dist) + '  ' + str(avg_triplet) + '\n')
    file.close()
    return
    

def valid(args, epoch, epochs, device, generator_partial, generator_complete, decoder_fused, data_loader_shapenet_val, best_dist, \
    result_dir, model_dir_partial, model_dir_complete, model_dir_decoder):
    '''
        description: valid the models for one epoch
        variable: args, epoch, epochs, device, model, data_loader_shapenet_val, best_dist, result_dir_PCN, model_dir_PCN
        return: best_dist
    '''
    
    generator_partial.eval()
    generator_complete.eval()
    decoder_fused.eval()
    
    total_dist = 0
    total_batch = 0
    for i, (partial_shapenet, ground_truth_fine, ground_truth_coarse)  in enumerate(data_loader_shapenet_val):
        if device:
            partial_shapenet = partial_shapenet.to(device)
            ground_truth_fine = ground_truth_fine.to(device)
            ground_truth_coarse = ground_truth_coarse.to(device)
            #partial_scannet = partial_scannet.to(device)
        
        feature_anchor = generator_partial(partial_shapenet)
        feature_positive = generator_complete(partial_shapenet)
        if args.loss == 'cosine':
            #归一化
            if args.normalize == True:
                feature_anchor = torch.nn.functional.normalize(feature_anchor, dim = 1)

        coarse_anchor, fine_anchor = decoder_fused(feature_anchor, feature_positive)
        dis_anchor = get_chamfer_dist_valid(coarse_anchor, fine_anchor, ground_truth_coarse, ground_truth_fine)
        total_dist += dis_anchor.item() * 10000
        total_batch += 1

        min_loss = dis_anchor.item()
        print('Valid:epoch:[{}/{}] batch {}, dis: {:.2f}'.format(epoch + 1, epochs, i+1, min_loss * 10000))    
    
    avg_dist = total_dist / total_batch
    file = open(result_dir, "a")
    file.write(str(avg_dist) + '\n')
    file.close()
    print('Valid:epoch:[{}/{}] total average dist: {:.2f}'.format(epoch + 1, epochs, avg_dist))
    if avg_dist < best_dist:
        best_dist = avg_dist
        torch.save(generator_partial.state_dict(), model_dir_partial)
        torch.save(generator_complete.state_dict(), model_dir_complete)
        torch.save(decoder_fused.state_dict(), model_dir_decoder)
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
        train(args, epoch, args.epochs, device, generator_partial, generator_complete, decoder_fused,\
            optimizer_generator_complete, optimizer_generator_partial, optimizer_fused, data_loader_shapenet_train, result_dir)
        best_dist = valid(args, epoch, args.epochs, device, generator_partial, generator_complete, decoder_fused, data_loader_shapenet_val, 14530529, \
            result_dir, model_dir_partial, model_dir_complete, model_dir_decoder)
