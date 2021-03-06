'''
Description: Util functions of triplet loss
Author:Charles Shen
Date:8/20/2020
'''

import random
import torch 
import torch.nn as nn
import torchvision
import constants

def random_sample(shapenet_partial, shapenet_fine, shapenet_coarse):
    '''
        description: random sample the negative sample, without coarse
        parameters: shapenet_partial(batch*num*num_points*3), shapenet_fine+coarse(batch * num_points * 3)
        return: shapenet_anchor, shapenet_positive, shapenet_negative((batch*num)*num_points*3), \
        positive_fine(正例和anchor用), positive_coarse, negative_fine(负例), negative_coarse
    '''
    #batch_size_scannet = scannet.size(0)
    anchor_list = []
    positive_list = []
    negative_list = []
    negative_fine_list = []
    negative_coarse_list = []

    batch_size = shapenet_partial.size(0)
    num_partial = shapenet_partial.size(1)

    anchor_torch = shapenet_partial.view(batch_size * num_partial, shapenet_partial.size(2), shapenet_partial.size(3))
    positive_fine = shapenet_fine.repeat(num_partial, 1, 1)    
    positive_coarse = shapenet_fine.repeat(num_partial, 1, 1)    

    for i in range(batch_size * num_partial):
        choice_positive = ((i % num_partial) + random.randint(1, num_partial - 1)) % num_partial + (i - i % num_partial)
        choice_negative = (((i // num_partial) + random.randint(1, batch_size - 1)) % batch_size)  * num_partial + random.randint(0, num_partial - 1)
        positive = anchor_torch[choice_positive].clone()
        negative = anchor_torch[choice_negative].clone()
        negative_fine_one = positive_fine[choice_negative].clone()
        negative_coarse_one = positive_coarse[choice_negative].clone()
        positive_list.append(positive)
        negative_list.append(negative)
        negative_fine_list.append(negative_fine_one)
        negative_coarse_list.append(negative_coarse_one)
    positive_torch = torch.stack(positive_list, dim = 0)
    negative_torch = torch.stack(negative_list, dim = 0)
    negative_fine = torch.stack(negative_fine_list, dim = 0)
    negative_coarse = torch.stack(negative_coarse_list, dim = 0)

    return anchor_torch, positive_torch, negative_torch, positive_fine, positive_coarse, negative_fine, negative_coarse

def random_sample_original(shapenet_partial, shapenet_fine):
    '''
        description: random sample the negative sample, without coarse, used in shapenet
        parameters: shapenet_partial, shapenet_fine
        return: negative_torch
    '''
    #batch_size_scannet = scannet.size(0)

    negative_list = []
    batch_size = shapenet_partial.size(0)

    for i in range(batch_size):
        negative_choice = random.randint(0, batch_size - 2)
        if negative_choice >= i:
            negative_choice += 1
        type_choice = random.randint(0, 1)
        if type_choice == 0:
            negative_list.append(shapenet_partial[negative_choice])
        else:
            negative_list.append(shapenet_fine[negative_choice])

    negative_torch = torch.stack(negative_list, dim = 0)
    return negative_torch

def random_sample_with_self(shapenet_partial, shapenet_fine):
    '''
        description: random sample the negative sample, without coarse, used in shapenet
        parameters: shapenet_partial, shapenet_fine
        return: anchor_torch, positive_torch, negative_torch
    '''
    #batch_size_scannet = scannet.size(0)
    anchor_list = []
    negative_list = []
    positive_list = []
    batch_size = shapenet_partial.size(0)

    for i in range(batch_size):
        indice = random.randint(0, 1)
        if indice == 0:
            anchor_list.append(shapenet_fine[i])
        else:
            anchor_list.append(shapenet_partial[i])
        indice = random.randint(0, 1)
        if indice == 0:
            positive_list.append(shapenet_fine[i])
        else:
            positive_list.append(shapenet_partial[i])
        negative_choice = random.randint(0, batch_size - 2)
        if negative_choice >= i:
            negative_choice += 1
        type_choice = random.randint(0, 1)
        if type_choice == 0:
            negative_list.append(shapenet_partial[negative_choice])
        else:
            negative_list.append(shapenet_fine[negative_choice])
    anchor_torch = torch.stack(anchor_list, dim = 0)
    positive_torch = torch.stack(positive_list, dim = 0)
    negative_torch = torch.stack(negative_list, dim = 0)
    return anchor_torch, positive_torch, negative_torch
