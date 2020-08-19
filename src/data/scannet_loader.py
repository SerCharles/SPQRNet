'''
Description:The data loader of shapenet
Author:Charles Shen
Data:8/18/2020
'''

import os
from plyfile import PlyData, PlyElement
import torch
import numpy as np
import pandas as pd
#import open3d
from utils.pointnet_util import farthest_point_sample, index_points
from torch.utils.data import Dataset


class ScanNetLoader(Dataset):
    '''
    参数：
        base_dir:数据集基本路径
        data_type:训练还是测试,字符串
        type_name:类型名称
        num_points:点云个数
    '''
    def __init__(self, base_dir, data_type, type_name, num_points = 2048):
        dir_partial = os.path.join(base_dir, data_type, type_name)
        model_names = os.listdir(dir_partial)
        partial_list = []
        for filename in model_names:
            try:
                plydata = PlyData.read(os.path.join(dir_partial, filename))  # 读取文件
                data = plydata.elements[0].data  # 读取数据
                data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
                data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
                property_names = data[0].dtype.names  # 读取property的名字
                for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
                    data_np[:, i] = data_pd[name]
                the_partial = torch.from_numpy(data_np[:, 0:3]).float()
                
                #采样
                #如果点数目不够，就复制
                current_num_points = the_partial.size(0)
                if(current_num_points != num_points):
                    #print(the_partial.size())
                    if(current_num_points < num_points):
                        repeat_time = num_points // current_num_points + 1
                        the_partial = the_partial.repeat(repeat_time, 1)
                    #print(the_partial.size())

                    #下采样
                    the_partial = the_partial.view(1, the_partial.size(0), the_partial.size(1))
                    the_partial = index_points(the_partial, farthest_point_sample(the_partial, num_points))
                    the_partial = the_partial.view(the_partial.size(1), the_partial.size(2))
                    #print(the_partial.size())

                    #存储数据，避免再次训练耗时很长
                    '''data_modified = open3d.geometry.PointCloud()
                    data_modified.points = open3d.utility.Vector3dVector(the_partial)
                    open3d.io.write_point_cloud(os.path.join(dir_partial, 'changed_' + filename), data_modified)'''
                partial_list.append(the_partial)
            except:
                print("Loading data " + filename + " failed")

        self.partial = torch.stack(partial_list, 0)

    def __len__(self):
        return self.partial.shape[0]

    def __getitem__(self, idx):
        return self.partial[idx]

