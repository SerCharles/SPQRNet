import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.generator import Generator
from models.decoder import Decoder
from data.scannet_loader import ScanNetLoader
from data.shapenet_loader import ShapeNetLoader
from utils.triplet_loss import random_sample
from torch.utils.data import DataLoader
import chamfer3D.dist_chamfer_3D
import constants
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()




def initialize():
    '''
        description: initialize models, data, optimizer
        variable: empty
        return: device, generator_partial, generator_complete, decoder, optimizer, data_loader_scannet_train, data_loader_scannet_val, data_loader_shapenet_train, data_loader_shapenet_val
    '''
    print('getting device...', end='')
    device = torch.device('cuda:8')
    torch.cuda.empty_cache()
    print('device got')

    print('Initialize model')
    generator_partial = Generator()
    generator_complete = Generator()
    decoder = Decoder()
    print('Getting dataset')
    dataset_scannet_train = ScanNetLoader(constants.scannet_place, 'train', constants.scannet_type_name, 2048)
    dataset_scannet_val = ScanNetLoader(constants.scannet_place, 'val', constants.scannet_type_name, 2048)
    dataset_shapenet_train = ShapeNetLoader(constants.shapenet_place, 'train', constants.shapenet_type_code, 512)
    dataset_shapenet_val = ShapeNetLoader(constants.shapenet_place, 'val', constants.shapenet_type_code, 512)
    print('Getting dataloader')
    data_loader_scannet_train = DataLoader(dataset_scannet_train, batch_size = constants.batch_size, shuffle = True, num_workers = 2)
    data_loader_scannet_val = DataLoader(dataset_scannet_val, batch_size = constants.batch_size, shuffle = True, num_workers = 2)
    data_loader_shapenet_train =  DataLoader(dataset_shapenet_train, batch_size = constants.batch_size, shuffle = True, num_workers = 2)
    data_loader_shapenet_val = DataLoader(dataset_shapenet_val, batch_size = constants.batch_size, shuffle = True, num_workers = 2)
    print('Data got!')
    if device:
        generator_partial = generator_partial.to(device)
        generator_complete = generator_complete.to(device)
        decoder = decoder.to(device)
    optimizer = torch.optim.Adam(D.parameters(), lr = constants.learning_rate)


    return device, generator_partial, generator_complete, decoder, optimizer, data_loader_scannet_train, data_loader_scannet_val, data_loader_shapenet_train, data_loader_shapenet_val



def train




def valid



if __name__ == "__main__":

