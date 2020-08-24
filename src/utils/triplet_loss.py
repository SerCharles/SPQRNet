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

def random_sample(shapenet_partial, shapenet_fine):
    '''
        description: random sample the negative sample, without coarse
        parameters: shapenet_partial, shapenet_fine
        return: shapenet_anchor, shapenet_positive, shapenet_negative
    '''
    batch_size = shapenet_partial.size(0)
    #batch_size_scannet = scannet.size(0)
    anchor_list = []
    positive_list = []
    negative_list = []

    select_range = 2 * batch_size - 2

    for i in range(batch_size):
        anchor = None
        positive = None
        negative = None
        choice_anchor = random.randint(0, 1)
        if choice_anchor == 0:
            anchor = shapenet_partial[i].clone()
        else:
            anchor = shapenet_fine[i].clone()
        choice_positive= random.randint(0, 1)
        if choice_positive == 0:
            positive = shapenet_partial[i].clone()
        else:
            positive = shapenet_fine[i].clone()
        
        choice = random.randint(0, select_range - 1)
        if choice < batch_size - 1:
            #其他partial
            if choice < i:
                negative = shapenet_partial[choice].clone()
                #print("partial", choice)
            else:
                negative = shapenet_partial[choice + 1].clone()
                #print("partial", choice + 1)
        else:
            if choice < i + batch_size - 1:
                negative = shapenet_fine[choice - batch_size + 1].clone()
                #print("fine", choice - batch_size + 1)
            else:
                negative = shapenet_fine[choice - batch_size + 2].clone()
                #print("fine", choice - batch_size + 2)
        anchor_list.append(anchor)
        positive_list.append(positive)
        negative_list.append(negative)
    anchor_torch = torch.stack(anchor_list, dim = 0)
    positive_torch = torch.stack(positive_list, dim = 0)
    negative_torch = torch.stack(negative_list, dim = 0)
    return anchor_torch, positive_torch, negative_torch
