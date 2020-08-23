'''
Description: constants
Author:Charles Shen
Date:8/17/2020
'''

#place of dataset
scannet_place = '/home/shenguanlin/scannet_extract'
shapenet_place = '/data1/xp/shapenet'
scannet_type_name = 'chair'
shapenet_type_code = '03001627'

#network details
batch_size = 8
learning_rate = 0.0001
num_epochs = 100


#multiple losses
times_triplet = 100
triplet_margin = 1.0

#saving place
result_path = '../result/base'
text_name_PCN = 'PCN.txt'
text_name = 'result.txt'
model_name_PCN = 'PCN.pt'
model_name_partial = 'partial.pt'
model_name_complete = 'complete.pt'
model_name_decoder = 'decoder.pt'