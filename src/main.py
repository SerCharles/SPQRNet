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
from utils.triplet_loss import random_sample
from torch.utils.data import DataLoader
import chamfer3D.dist_chamfer_3D
import constants
from initialize import initialize, build_args
from utils.triplet_loss import random_sample
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()




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
        negative_examples = random_sample(partial_shapenet, ground_truth_fine)
        feature_partial = generator_partial(partial_shapenet)
        feature_positive = generator_complete(ground_truth_fine)
        feature_negative = generator_complete(negative_examples)

        #loss
        if args.loss == 'triplet':
            triplet_loss_function = torch.nn.TripletMarginLoss(margin = args.margin_triplet, p = 2)
            triplet_loss = triplet_loss_function(feature_partial, feature_positive, feature_negative)
            the_times_triplet = args.times_triplet
        elif args.loss == 'cosine':
            cosine_loss_function = torch.nn.CosineEmbeddingLoss(margin = args.margin_cosine)
            y_positive = torch.ones(feature_partial.size(0))
            y_negative = - torch.ones(feature_partial.size(0))
            if device:
                y_positive = y_positive.to(device)
                y_negative = y_negative.to(device)
            triplet_loss = cosine_loss_function(feature_partial, feature_positive, y_positive) + \
                cosine_loss_function(feature_partial, feature_negative, y_negative)
            the_times_triplet = args.times_cosine

        #reconstruction loss
        coarse, fine = decoder(feature_partial)
        dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
        dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
        dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, ground_truth_coarse)
        dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
        dis = dis_fine + 0.5 * dis_coarse
            
        total_loss = triplet_loss * (the_times_triplet / 10000) + dis
        total_dist += dis.item() * 10000
        total_triplet += triplet_loss.item()
        total_batch += 1


        optimizer_generator_complete.zero_grad()
        optimizer_generator_partial.zero_grad()
        optimizer_decoder.zero_grad()
        total_loss.backward()
        optimizer_generator_complete.step()
        optimizer_generator_partial.step()
        optimizer_decoder.step()

        print('Train:epoch:[{}/{}] batch {}, dis: {:.2f}, triplet: {:.6f}'.format(epoch + 1, epochs, i+1, dis.item() * 10000, triplet_loss.item()))
    
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
            
        feature_partial = generator_partial(partial_shapenet)
        coarse, fine = decoder(feature_partial)
        dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
        dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
        dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, ground_truth_coarse)
        dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
        dis = dis_fine + 0.5 * dis_coarse

        total_dist += dis.item() * 10000
        total_batch += 1

        print('Valid:epoch:[{}/{}] batch {}, dis: {:.2f}'.format(epoch + 1, epochs, i+1, dis.item() * 10000))    
    
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
