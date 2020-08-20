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
import chamfer3D.dist_chamfer_3D
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
    G = Decoder()
    D = Discriminator(in_size = 6144)
    print('Getting dataset')
    dataset_shapenet_train = ShapeNetLoader(constants.shapenet_place, 'train', constants.shapenet_type_code, 512)
    print('Getting dataloader')
    data_loader_shapenet_train =  DataLoader(dataset_shapenet_train, batch_size = constants.batch_size_shapenet, shuffle = True, num_workers = 2)
    print('Data got!')
    if device:
        G.to(device)
        D.to(device)
    criterion = nn.BCELoss()
    optimizer_d = torch.optim.Adam(D.parameters(), lr = constants.d_learning_rate)
    optimizer_g = torch.optim.Adam(G.parameters(), lr = constants.decoder_learning_rate)

    return device, G, D, criterion, optimizer_g, optimizer_d, data_loader_shapenet_train



def train_GAN(device, G, D, criterion, optimizer_g, optimizer_d, data_loader_shapenet_train):
    '''
        description: train GAN
        variable: device, G, D, criterion, optimizer_g, optimizer_d, data_loader_scannet_train, data_loader_shapenet_train
        return: G,D
    '''
    print("Training GAN", "*" * 100)
    for epoch in range(constants.num_epochs_GAN):
        
        print("Training discriminator")
        for d_index in range(constants.d_steps):
            # 1. Train D on real+fake
            D.zero_grad()
            D.train()
            G.eval()


                
            for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader_shapenet_train):
                if device:
                    ground_truth_fine = ground_truth_fine.to(device)
                    ground_truth_coarse = ground_truth_coarse.to(device)
                raw_data = torch.randn(ground_truth_fine.size(0), 1024).to(device)
                coarse, fine = G(raw_data)
                
                decision_real = D(torch.reshape(ground_truth_fine,(ground_truth_fine.size(0), 6144)))
                decision_fake = D(torch.reshape(fine,(fine.size(0), 6144)))
                decision_true = torch.ones([decision_real.size(0),1])
                decision_true = decision_true.to(device)
                decision_false = torch.zeros([decision_fake.size(0),1])
                decision_false = decision_false.to(device)
                d_real_error = criterion(decision_real, decision_true)  # ones = true
                d_real_error.backward() # compute/store gradients, but don't change params
                d_fake_error = criterion(decision_fake, decision_false)  # zeros = fake
                d_fake_error.backward()
                optimizer_d.step()
                #print(D.gradients())
                mean_error = (torch.mean(d_real_error) + torch.mean(d_fake_error)) / 2
                print('Discriminator:epoch:[{}/{}] batch {}, error: {}'.format(epoch + 1, constants.num_epochs_GAN, i+1, mean_error))
        
        print("Training generator")
        for g_index in range(constants.g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()
            G.train()
            D.eval()
            for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader_shapenet_train):
                if device:
                    ground_truth_fine = ground_truth_fine.to(device)
                    ground_truth_coarse = ground_truth_coarse.to(device)
                raw_data = torch.randn(ground_truth_fine.size(0), 1024).to(device)
                coarse, fine = G(raw_data)
                dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
                dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
                decision_fake = D(torch.reshape(fine,(fine.size(0), 6144)))
                decision_true = torch.ones([decision_fake.size(0),1])
                decision_true = decision_true.to(device)
                g_error_a = criterion(decision_fake, decision_true)  # Train G to pretend it's genuine
                g_error_a.backward()
                optimizer_g.step()  # Only optimizes G's parameters
                #print(G.gradients())
                mean_error = torch.mean(g_error_a)
                print('Generator:epoch:[{}/{}] batch {}, error: {}, dis: {}'.format(epoch + 1, constants.num_epochs_GAN, i+1, mean_error, dis_fine))
        for name, parms in G.named_parameters():	
            print('-->name:', name, ' -->grad_value:', torch.norm(parms.grad,2))
    
    return G, D






if __name__ == "__main__":
    device, G, D, criterion, optimizer_g, optimizer_d, data_loader_shapenet_train = initialize()
    G, D = train_GAN(device, G, D, criterion, optimizer_g, optimizer_d, data_loader_shapenet_train)
