import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.generator import Generator
from models.discriminator import Discriminator
from models.decoder import Decoder
from data.scannet_loader import ScanNetLoader
from data.shapenet_loader import ShapeNetLoader
from torch.utils.data import DataLoader
import constants
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()




def initialize():
    '''
        description: initialize models, data, optimizer
        variable: empty
        return: device, G, D, decoder, criterion, optimizer_g, optimizer_d, optimizer_decoder, data_loader_scannet_train, data_loader_scannet_val, data_loader_shapenet_train, data_loader_shapenet_val
    '''
    print('getting device...', end='')
    device = torch.device('cuda:8')
    torch.cuda.empty_cache()
    print('device got')

    print('Initialize model')
    G = Generator()
    D = Discriminator()
    decoder = Decoder()
    print('Getting dataset')
    dataset_scannet_train = ScanNetLoader(constants.scannet_place, 'train', constants.scannet_type_name, 2048)
    dataset_scannet_val = ScanNetLoader(constants.scannet_place, 'val', constants.scannet_type_name, 2048)
    dataset_shapenet_train = ShapeNetLoader(constants.shapenet_place, 'train', constants.shapenet_type_code, 512)
    dataset_shapenet_val = ShapeNetLoader(constants.shapenet_place, 'val', constants.shapenet_type_code, 512)
    print('Getting dataloader')
    data_loader_scannet_train = DataLoader(dataset_scannet_train, batch_size = constants.batch_size_scannet, shuffle = True, num_workers = 2)
    data_loader_scannet_val = DataLoader(dataset_scannet_val, batch_size = constants.batch_size_scannet, shuffle = True, num_workers = 2)
    data_loader_shapenet_train =  DataLoader(dataset_shapenet_train, batch_size = constants.batch_size_shapenet, shuffle = True, num_workers = 2)
    data_loader_shapenet_val = DataLoader(dataset_shapenet_val, batch_size = constants.batch_size_shapenet, shuffle = True, num_workers = 2)
    print('Data got!')
    if device:
        G.to(device)
        D.to(device)
        decoder.to(device)
    criterion = nn.BCELoss()
    optimizer_d = torch.optim.Adam(D.parameters(), lr = constants.d_learning_rate)
    optimizer_g = torch.optim.Adam(G.parameters(), lr = constants.g_learning_rate)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr = constants.decoder_learning_rate)

    return device, G, D, decoder, criterion, optimizer_g, optimizer_d, optimizer_decoder, data_loader_scannet_train, data_loader_scannet_val, data_loader_shapenet_train, data_loader_shapenet_val


def train_GAN(device, G, D, criterion, optimizer_g, optimizer_d, data_loader_scannet_train, data_loader_shapenet_train):
    '''
        description: train GAN
        variable: device, G, D, criterion, optimizer_g, optimizer_d, data_loader_scannet_train, data_loader_shapenet_train
        return: G,D
    '''
    print("Training GAN", "*" * 100)
    for epoch in range(constants.num_epochs_GAN):
        print("this is epoch ", epoch)
        for d_index in range(constants.d_steps):
            # 1. Train D on real+fake
            D.zero_grad()
            D.train()
            G.eval()
            for i, ((data_shapenet, _, _),(data_scannet)) in enumerate(zip(data_loader_shapenet_train, data_loader_scannet_train)):
                if device:
                    data_shapenet = data_shapenet.to(device)
                    data_scannet = data_scannet.to(device)
                
                decision_shapenet, decision_scannet = G(data_shapenet, data_scannet)
                d_real_error = criterion(decision_shapenet, Variable(torch.ones([decision_shapenet.size(0),1])))  # ones = true
                d_real_error.backward() # compute/store gradients, but don't change params
                d_fake_error = criterion(decision_scannet, Variable(torch.zeros([decision_scannet.size(0),1])))  # zeros = fake
                d_fake_error.backward()
                optimizer_d.step()

        for g_index in range(constants.g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()
            G.train()
            D.eval()
            for i, ((data_shapenet, _, _),(data_scannet)) in enumerate(zip(data_loader_shapenet_train, data_loader_scannet_train)):
                if device:
                    data_shapenet = data_shapenet.to(device)
                    data_scannet = data_scannet.to(device)
                
                decision_shapenet, decision_scannet = G(data_shapenet, data_scannet)
                g_error_a = criterion(decision_scannet, Variable(torch.ones([decision_scannet.size(0),1])))  # Train G to pretend it's genuine
                g_error_a.backward()
                g_error_b = criterion(decision_shapenet, Variable(torch.zeros([decision_shapenet.size(0),1])))  # Train G to pretend it's genuine
                g_error_b.backward()
                optimizer_g.step()  # Only optimizes G's parameters
    return G, D

def train_Decoder(device, G, decoder, optimizer_decoder, data_loader_shapenet_train):
    '''
        description: train Decoder
        variable: device, G, decoder, optimizer_decoder, data_loader_shapenet_train
        return: decoder
    '''
    best_train_dis = 114514
    print("Training Decoder", "*" * 100)
    decoder.train()
    G.eval()
    for epoch in range(constants.num_epochs_decoder):
        for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader_shapenet_train):
            if device:
                the_data = the_data.to(device)
                ground_truth_fine = ground_truth_fine.to(device)
                ground_truth_coarse = ground_truth_coarse.to(device)
            the_feature = G.get_x_result(the_data)
            coarse, fine = decoder(the_feature)
            dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
            dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
            dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, ground_truth_coarse)
            dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
            dis = dis_fine + 0.5 * dis_coarse
            optimizer_decoder.zero_grad()
            dis.backward()
            optimizer_decoder.step()
            print('epoch:[{}/{}] batch {}, dis: {}'.format(epoch, epochs, i+1, dis.item() * 10000))
    return decoder


def valid(device, G, decoder, data_loader_scannet_val, data_loader_shapenet_val):
    '''
        description: valid entire network
        variable: device, G, decoder, data_loader_scannet_val, data_loader_shapenet_val
        return: empty
    '''
    print("Validing", "*" * 100)
    G.eval()
    decoder.eval()
    for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader_shapenet_val):
        if device:
            the_data = the_data.to(device)
            ground_truth_fine = ground_truth_fine.to(device)
            ground_truth_coarse = ground_truth_coarse.to(device)
        the_feature = G.get_x_result(the_data)
        coarse, fine = decoder(the_feature)
        dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
        dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
        dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, ground_truth_coarse)
        dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
        dis = dis_fine + 0.5 * dis_coarse
        print('epoch:[{}/{}] batch {}, dis: {}'.format(epoch, epochs, i+1, dis.item() * 10000))





if __name__ == "__main__":
    device, G, D, decoder, criterion, optimizer_g, optimizer_d, optimizer_decoder, data_loader_scannet_train, data_loader_scannet_val, data_loader_shapenet_train, data_loader_shapenet_val = initialize()
    G, D = train_GAN(device, G, D, criterion, optimizer_g, optimizer_d, data_loader_scannet_train, data_loader_shapenet_train)
    decoder = train_Decoder(device, G, decoder, optimizer_decoder, data_loader_shapenet_train)
    valid(device, G, decoder, data_loader_scannet_val, data_loader_shapenet_val)
