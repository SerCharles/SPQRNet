import numpy as np
import time
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
from utils.triplet_loss import random_sample
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()



def initialize():
    '''
        description: initialize models, data, optimizer
        variable: empty
        return: device, generator_partial, generator_complete, decoder, optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder ,\
                data_loader_shapenet_train, data_loader_shapenet_val
    '''
    print('getting device...', end='')
    device = torch.device('cuda:8')
    torch.cuda.empty_cache()
    print('device got')

    print('Initialize model')
    generator_partial = Generator(project_size = 1024)
    generator_complete = Generator(project_size = 1024)
    decoder = Decoder()
    print('Getting dataset')
    #dataset_scannet_train = ScanNetLoader(constants.scannet_place, 'train', constants.scannet_type_name, 2048)
    #dataset_scannet_val = ScanNetLoader(constants.scannet_place, 'val', constants.scannet_type_name, 2048)
    dataset_shapenet_train = ShapeNetLoader(constants.shapenet_place, 'train', constants.shapenet_type_code, 512)
    dataset_shapenet_val = ShapeNetLoader(constants.shapenet_place, 'val', constants.shapenet_type_code, 512)
    print('Getting dataloader')
    #data_loader_scannet_train = DataLoader(dataset_scannet_train, batch_size = constants.batch_size, shuffle = True, num_workers = 2)
    #data_loader_scannet_val = DataLoader(dataset_scannet_val, batch_size = constants.batch_size, shuffle = True, num_workers = 2)
    data_loader_shapenet_train =  DataLoader(dataset_shapenet_train, batch_size = constants.batch_size, shuffle = True, num_workers = 2)
    data_loader_shapenet_val = DataLoader(dataset_shapenet_val, batch_size = constants.batch_size, shuffle = True, num_workers = 2)
    print('Data got!')
    if device:
        generator_partial.to(device)
        generator_complete.to(device)
        decoder.to(device)
    optimizer_generator_partial = torch.optim.Adam(generator_partial.parameters(), lr = constants.learning_rate)
    optimizer_generator_complete = torch.optim.Adam(generator_complete.parameters(), lr = constants.learning_rate)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr = constants.learning_rate)


    return device, generator_partial, generator_complete, decoder, optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder,\
           data_loader_shapenet_train, data_loader_shapenet_val



def train(epoch, epochs, device, generator_partial, generator_complete, decoder, \
optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, data_loader_shapenet_train):
    '''
        description: train the models for one epoch
        variable: epoch, epochs, device, generator_partial, generator_complete, decoder, \
        optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, data_loader_shapenet_train
        return: generator_partial, generator_complete, decoder
    '''
    generator_partial.train()
    generator_complete.train()
    decoder.train()

    for i, (partial_shapenet, ground_truth_fine, ground_truth_coarse)  in enumerate(data_loader_shapenet_train):
            if device:
                partial_shapenet = partial_shapenet.to(device)
                ground_truth_fine = ground_truth_fine.to(device)
                ground_truth_coarse = ground_truth_coarse.to(device)
                #partial_scannet = partial_scannet.to(device)
            
            #triplet loss
            negative_examples = random_sample(partial_shapenet, ground_truth_fine)
            feature_partial = generator_partial(partial_shapenet)
            feature_positive = generator_complete(ground_truth_fine)
            feature_negative = generator_complete(negative_examples)
            triplet_loss_function = torch.nn.TripletMarginLoss(margin = 1.0, p = 2)
            triplet_loss = triplet_loss_function(feature_partial, feature_positive, feature_negative)


            #reconstruction loss
            coarse, fine = decoder(feature_partial)
            dis_fine1, dis_fine2, _, _ = chamLoss(fine, ground_truth_fine)
            dis_fine = torch.mean(dis_fine1) + torch.mean(dis_fine2)
            dis_coarse1, dis_coarse2, _, _ = chamLoss(coarse, ground_truth_coarse)
            dis_coarse = torch.mean(dis_coarse1) + torch.mean(dis_coarse2)
            dis = dis_fine + 0.5 * dis_coarse
            
            total_loss = triplet_loss + constants.times_reconstruction * dis

            optimizer_generator_complete.zero_grad()
            optimizer_generator_partial.zero_grad()
            optimizer_decoder.zero_grad()
            total_loss.backward()
            optimizer_generator_complete.step()
            optimizer_generator_partial.step()
            optimizer_decoder.step()

            print('Train:epoch:[{}/{}] batch {}, dis: {:.2f}, triplet: {:.6f}'.format(epoch + 1, epochs, i+1, dis.item() * 10000, triplet_loss.item()))
    return generator_partial, generator_complete, decoder

def valid(epoch, epochs, device, generator_partial, generator_complete, decoder, data_loader_shapenet_val, best_dist):
    '''
        description: valid the models for one epoch
        variable: epoch, epochs, device, generator_partial, generator_complete, decoder, data_loader_shapenet_val, best_dist
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
    print('Valid:epoch:[{}/{}] total average dist: {:.2f}'.format(epoch + 1, epochs, avg_dist))
    if avg_dist < best_dist:
        best_dist = avg_dist
        torch.save(generator_partial.state_dict(), constants.model_path_partial)
        torch.save(generator_complete.state_dict(), constants.model_path_complete)
        torch.save(decoder.state_dict(), constants.model_path_decoder)
    return best_dist



if __name__ == "__main__":
    device, generator_partial, generator_complete, decoder, optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder,\
           data_loader_shapenet_train, data_loader_shapenet_val= initialize()
    for epoch in range(constants.num_epochs):
        generator_partial, generator_complete, decoder = train(epoch, constants.num_epochs, device, generator_partial, generator_complete, decoder, \
optimizer_generator_complete, optimizer_generator_partial, optimizer_decoder, data_loader_shapenet_train)
        best_dist = valid(epoch, constants.num_epochs, device, generator_partial, generator_complete, decoder, data_loader_shapenet_val, 14530529)
    file = open(constants.result_path, "w")
    file.write(str(best_dist))
    file.close()