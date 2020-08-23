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
        description: random sample the negative sample
        parameters: shapenet_partial, shapenet_fine(positive)
        return: negative sample
    '''
    batch_size = shapenet_partial.size(0)
    #batch_size_scannet = scannet.size(0)
    select_range = 2 * batch_size - 2
    negative_list = []
    for i in range(batch_size):
        negative = None
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

        negative_list.append(negative)
    negative_torch = torch.stack(negative_list, dim = 0)
    return negative_torch

