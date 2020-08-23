'''
Description: Initializing the arg parser and the models, optimizers, dataloaders...etc
Author:Charles Shen
Date:8/22/2020
'''

import argparse
import numpy as np
import time
import os
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
import constants


def build_args():
    '''
        description: initialize arg parser
        variable: no
        return: args
    '''
    parser = argparse.ArgumentParser(description = 'PCN Network')
    parser.add_argument('--epochs', type = int, default = constants.num_epochs, help='upper epoch limit')
    parser.add_argument('--batch_size', type = int, default = constants.batch_size, help='set batch size')
    parser.add_argument('--lr', type = float, default = constants.learning_rate, help='set batch size')

    parser.add_argument('--seed', type = int, default = 1234, help='set random seed')
    parser.add_argument('--cuda',  type = bool, default = True, help = 'use GPU or not')
    parser.add_argument('--gpu_id', type = int, default = 0, help = 'GPU device id used')

    parser.add_argument('--result_path', type = str, default = constants.result_path, help = 'the path folder to save your result')
    parser.add_argument('--scannet_path', type = str, default = constants.scannet_place, help = 'the path folder of scannet')
    parser.add_argument('--shapenet_path', type = str, default = constants.shapenet_place, help = 'the path folder of shapenet')


    parser.add_argument('--times_triplet', type = int, default = constants.times_triplet, help = 'The ratio of triplet loss(after * 10000)')
    parser.add_argument('--margin', type = float, default = constants.triplet_margin, help = 'The margin of triplet loss')

    args = parser.parse_args()

    # Set the random seed manually for reproducibility.

    return args

def initialize(args):
    '''
        description: initialize models, data, optimizer
        variable: args
        return: device, generator_partial, generator_complete, decoder, pcn, optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, optimizer_PCN\
           data_loader_shapenet_train, data_loader_shapenet_val, result_dir_PCN, result_dir, model_dir_PCN, model_dir_partial, model_dir_complete, model_dir_decoder
    '''
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

    print('Initialize model')
    generator_partial = Encoder()
    generator_complete = Encoder()
    decoder = Decoder()
    pcn = PCN()
    if device:
        generator_partial.to(device)
        generator_complete.to(device)
        decoder.to(device)
        pcn.to(device)

    print('Getting optimizer')
    optimizer_generator_partial = torch.optim.Adam(generator_partial.parameters(), lr = args.lr)
    optimizer_generator_complete = torch.optim.Adam(generator_complete.parameters(), lr = args.lr)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr = args.lr)
    optimizer_PCN = torch.optim.Adam(pcn.parameters(), lr = args.lr)

    print('Getting result dir')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    result_dir_PCN = os.path.join(args.result_path, constants.text_name_PCN)
    result_dir = os.path.join(args.result_path, constants.text_name)
    model_dir_PCN = os.path.join(args.result_path, constants.model_name_PCN)
    model_dir_partial = os.path.join(args.result_path, constants.model_name_partial)
    model_dir_complete = os.path.join(args.result_path, constants.model_name_complete)
    model_dir_decoder = os.path.join(args.result_path, constants.model_name_decoder)


    print('Getting dataset')
    #dataset_scannet_train = ScanNetLoader(args.scannet_path, 'train', constants.scannet_type_name, 2048)
    #dataset_scannet_val = ScanNetLoader(args.scannet_path, 'val', constants.scannet_type_name, 2048)
    dataset_shapenet_train = ShapeNetLoader(args.shapenet_path, 'train', constants.shapenet_type_code, 512)
    dataset_shapenet_val = ShapeNetLoader(args.shapenet_path, 'val', constants.shapenet_type_code, 512)
    print('Getting dataloader')
    #data_loader_scannet_train = DataLoader(dataset_scannet_train, batch_size = args.batch_size, shuffle = True, num_workers = 2)
    #data_loader_scannet_val = DataLoader(dataset_scannet_val, batch_size = args.batch_size, shuffle = True, num_workers = 2)
    data_loader_shapenet_train =  DataLoader(dataset_shapenet_train, batch_size = args.batch_size, shuffle = True, num_workers = 2)
    data_loader_shapenet_val = DataLoader(dataset_shapenet_val, batch_size = args.batch_size, shuffle = True, num_workers = 2)
    print('Data got!')


    return device, generator_partial, generator_complete, decoder, pcn, optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, optimizer_PCN, \
           data_loader_shapenet_train, data_loader_shapenet_val, result_dir_PCN, result_dir, model_dir_PCN, model_dir_partial, model_dir_complete, model_dir_decoder
