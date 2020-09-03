'''
Description: Main program used in train and testing my network
Author:Charles Shen
Date:8/22/2020
'''

import numpy as np
import time
import os
import argparse
import torch
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
from utils.triplet_loss import random_sample
from utils.common import get_triplet_loss, get_chamfer_dist_train, get_chamfer_dist_valid, step_weight


def train(args, epoch, epochs, device, generator_partial, generator_complete, decoder, \
optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, data_loader_shapenet_train, result_dir):
    '''
        description: train the models for one epoch
        variable: args, epoch, epochs, device, generator_partial, generator_complete, decoder, \
            optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, data_loader_shapenet_train, result_dir
        return: generator_partial, generator_complete, decoder
    '''
    generator_partial.train()
    generator_complete.train()
    decoder.train()


    total_dist = 0
    total_triplet = 0
    total_batch = 0
    for i, (partial_shapenet, ground_truth_fine, ground_truth_coarse)  in enumerate(data_loader_shapenet_train):
        if device:
            partial_shapenet = partial_shapenet.to(device)
            ground_truth_fine = ground_truth_fine.to(device)
            ground_truth_coarse = ground_truth_coarse.to(device)
            #partial_scannet = partial_scannet.to(device)
        
        #medium feature
        if args.dataset == 'complete':
            batch_size = partial_shapenet.size(0)
            num_partial = partial_shapenet.size(1)
            anchor_examples, positive_examples, negative_examples, positive_fine_gt, positive_coarse_gt, negative_fine_gt, negative_coarse_gt\
             = random_sample(partial_shapenet, ground_truth_fine, ground_truth_coarse)
            feature_anchor = generator_partial(anchor_examples)
            feature_positive = generator_complete(positive_examples)
            feature_negative = generator_complete(negative_examples)
        else:            
            negative_examples = random_sample_original(partial_shapenet, ground_truth_fine)
            feature_anchor = generator_partial(partial_shapenet)
            feature_positive = generator_complete(ground_truth_fine)
            feature_negative = generator_complete(negative_examples)

        triplet_loss, the_times_triplet, feature_anchor, feature_positive, feature_negative = \
            get_triplet_loss(feature_anchor, feature_positive, feature_negative, args, device)


        #reconstruction loss
        #anchor
        weight_triplet = step_weight(args, epoch, epochs, the_times_triplet)

        if args.dataset == 'complete':
            coarse_anchor, fine_anchor = decoder(feature_anchor)
            dis_anchor = get_chamfer_dist_train(coarse_anchor, fine_anchor, positive_coarse_gt, positive_fine_gt)

            coarse_positive, fine_positive = decoder(feature_positive)
            dis_positive = get_chamfer_dist_train(coarse_positive, fine_positive, positive_coarse_gt, positive_fine_gt)
            
            coarse_negative, fine_negative = decoder(feature_negative)
            dis_negative = get_chamfer_dist_train(coarse_negative, fine_negative, negative_coarse_gt, negative_fine_gt)

            total_loss = triplet_loss * (weight_triplet / 10000) + \
                (dis_anchor * args.weight_anchor + dis_positive *args.weight_positive + dis_negative * args.weight_negative)
        else:
            coarse_anchor, fine_anchor = decoder(feature_anchor)
            dis_anchor = get_chamfer_dist_train(coarse_anchor, fine_anchor, ground_truth_fine, ground_truth_coarse)
            total_loss = triplet_loss * (weight_triplet / 10000) + dis_anchor

        if args.dataset == 'complete':
            total_dist += min(dis_anchor.item(), dis_positive.item()) * 10000
        else:
            total_dist += dis_anchor.item() * 10000
        total_triplet += triplet_loss.item()
        total_batch += 1


        optimizer_generator_complete.zero_grad()
        optimizer_generator_partial.zero_grad()
        optimizer_decoder.zero_grad()
        total_loss.backward()
        optimizer_generator_complete.step()
        optimizer_generator_partial.step()
        optimizer_decoder.step()

        if args.dataset == 'complete':
            min_loss = min(dis_anchor.item(), dis_positive.item())
        else:
            min_loss = dis_anchor.item()
        print('Train:epoch:[{}/{}] batch {}, dis: {:.2f}, triplet: {:.6f}'.format(epoch + 1, epochs, i+1, min_loss * 10000, triplet_loss.item()))
    
    avg_dist = total_dist / total_batch
    avg_triplet = total_triplet / total_batch
    file = open(result_dir, "a")
    file.write(str(avg_dist) + '  ' + str(avg_triplet) + '\n')
    file.close()

    return generator_partial, generator_complete, decoder

def valid(args, epoch, epochs, device, generator_partial, generator_complete, decoder, data_loader_shapenet_val, best_dist, \
    result_dir, model_dir_partial, model_dir_complete, model_dir_decoder):
    '''
        description: valid the models for one epoch
        variable:args, epoch, epochs, device, generator_partial, generator_complete, decoder, data_loader_shapenet_val, best_dist, \
            result_dir, model_dir_partial, model_dir_complete, model_dir_decoder
        return: best_dist
    '''
    
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
        if args.dataset == 'complete':
            min_loss = min(dis_anchor.item(), dis_positive.item())
        else:
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
        torch.save(decoder.state_dict(), model_dir_decoder)
    return best_dist



if __name__ == "__main__":
    args = build_args()
    device, generator_partial, generator_complete, decoder, pcn, optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, optimizer_PCN, \
        data_loader_shapenet_train, data_loader_shapenet_val, result_dir_PCN, result_dir, model_dir_PCN, model_dir_partial, model_dir_complete, model_dir_decoder = initialize(args)
    try:
        os.remove(result_dir)
    except:
        pass


    for epoch in range(constants.num_epochs):
        generator_partial, generator_complete, decoder = train(args, epoch, args.epochs, device, generator_partial, generator_complete, decoder, \
            optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, data_loader_shapenet_train, result_dir)
        best_dist = valid(args, epoch, args.epochs, device, generator_partial, generator_complete, decoder, data_loader_shapenet_val, 14530529, \
            result_dir, model_dir_partial, model_dir_complete, model_dir_decoder)
