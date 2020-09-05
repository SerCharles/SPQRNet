'''
Description: Util functions
Author:Charles Shen
Date:8/20/2020
'''

import numpy as np
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import chamfer3D.dist_chamfer_3D
import constants
from models.encoder import Encoder
from models.decoder import Decoder
from models.PCN import PCN
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()


def get_chamfer_dist_train(coarse, fine, coarse_gt, fine_gt):
    '''
        description: get chamfer distance(train)
        variable: coarse, fine, coarse_gt, fine_gt
        return: dis
    '''
    dis_fine1, dis_fine2, _, _ = chamLoss(fine, fine_gt)
    dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
    dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, coarse_gt)
    dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
    dis = dis_fine + dis_coarse
    return dis

def get_chamfer_dist_valid(coarse, fine, coarse_gt, fine_gt):
    '''
        description: get chamfer distance(valid)
        variable: coarse, fine, coarse_gt, fine_gt
        return: dis_fine
    '''
    dis_fine1, dis_fine2, _, _ = chamLoss(fine, fine_gt)
    dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
    return dis_fine

def get_triplet_loss(feature_anchor, feature_positive, feature_negative, args, device):
    '''
        description: calculate the triplet loss between features of anchor, positive and negative
        variable: feature_anchor, feature_positive, feature_negative, args, device
        return: triplet_loss, the_times_triplet, feature_anchor, feature_positive, feature_negative
    '''
    if args.loss == 'triplet':
        triplet_loss_function = torch.nn.TripletMarginLoss(margin = args.margin_triplet, p = 2)
        triplet_loss = triplet_loss_function(feature_anchor, feature_positive, feature_negative)
        the_times_triplet = args.times_triplet
    elif args.loss == 'cosine':
        #归一化
        if args.normalize == True:
            feature_anchor = torch.nn.functional.normalize(feature_anchor, dim = 1)
            feature_positive = torch.nn.functional.normalize(feature_positive, dim = 1)
            feature_negative = torch.nn.functional.normalize(feature_negative, dim = 1)

        cosine_loss_function = torch.nn.CosineEmbeddingLoss(margin = args.margin_cosine)
        y_positive = torch.ones(feature_anchor.size(0))
        y_negative = - torch.ones(feature_anchor.size(0))
        if device:
            y_positive = y_positive.to(device)
            y_negative = y_negative.to(device)
        triplet_loss = cosine_loss_function(feature_anchor, feature_positive, y_positive) + \
            cosine_loss_function(feature_anchor, feature_negative, y_negative)
        the_times_triplet = args.times_cosine
    return triplet_loss, the_times_triplet, feature_anchor, feature_positive, feature_negative

def step_weight(args, epoch, epochs, the_times_triplet):
    '''
        description: change the weight of triplet loss by epoch steps
        variable: args, epoch, epochs, the_times_triplet
        return: weight_triplet
    '''
    if args.step == False:
        return the_times_triplet
    else:
        if epoch <= epochs * 0.2:
            current_times_triplet = 0
        elif epoch <= epochs * 0.4:
            current_times_triplet = the_times_triplet * 0.25
        elif epoch <= epochs * 0.6:
            current_times_triplet = the_times_triplet * 0.5
        elif epoch <= epochs * 0.8:
            current_times_triplet = the_times_triplet * 0.75
        else:
            current_times_triplet = the_times_triplet
    return current_times_triplet

def show_parameters(model):
    '''
        description: print the parameters and grads of a model
        variable: model
        return: empty
    '''
    for name, parameter in model.named_parameters():
        print(torch.norm(parameter))
        print(torch.norm(parameter.grad))