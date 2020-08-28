import torch
from data.shapenet_loader import load_shapenet_all
from models.PCN import PCN
from torch.utils.data import DataLoader
import os
import open3d
import numpy as np
import constants

#DATA_PATH = '/home/shenguanlin/shapenet/train'
DATA_PATH = 'E:\\dataset\\shapenet'
TYPE_CODE = '04256520'
BATCH_SIZE = 8



    
def train():
    model = PCN()
    model.load_state_dict(torch.load('..\\result\\best.pt',  map_location='cpu'))
    model.eval()
    
    print('getting dataset')
    dataset_shapenet_train, dataset_shapenet_val = load_shapenet_all(DATA_PATH, constants.types, 512)    
    print('getting dataloader')
    data_loader = DataLoader(dataset_shapenet_val, batch_size = constants.batch_size, shuffle=True, num_workers=2)
    print('data got!')

    for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):

        result_coarse, result_fine = model(the_data)

        result_fine_np = result_fine[0].detach().numpy()

        result_fine_pcd = open3d.geometry.PointCloud()

        # From numpy to Open3D
        result_fine_pcd.points = open3d.utility.Vector3dVector(result_fine_np)

        #open3d.visualization.draw_geometries([result_fine_pcd])
        open3d.io.write_point_cloud('..\\result\\pointcloud\\result_fine_' + str(i) + '.ply', result_fine_pcd)
        return


def show():
    print('getting dataset')
    dataset = MyDataset(DATA_PATH, TYPE_CODE)
    print('getting dataloader')
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('data got!')

    for i, (the_data, ground_truth_fine, ground_truth_coarse) in enumerate(data_loader):

        ground_truth_fine_np = ground_truth_fine[0].detach().numpy()

        ground_truth_fine_pcd = open3d.geometry.PointCloud()

        # From numpy to Open3D
        ground_truth_fine_pcd.points = open3d.utility.Vector3dVector(ground_truth_fine_np)

        open3d.visualization.draw_geometries([ground_truth_fine_pcd])
        open3d.io.write_point_cloud('ground_truth_fine.ply', ground_truth_fine_pcd)
        return

if __name__ == "__main__":
    train()
    #show()