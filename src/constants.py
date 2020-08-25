'''
Description: constants
Author:Charles Shen
Date:8/17/2020
'''

#place of dataset
scannet_place = '/home/shenguanlin/scannet_extract'
shapenet_place = '/data1/xp/shapenet'
shapenet_complete_place = '/home/shenguanlin/shapenet_completion'
type_name = 'chair'
type_code = '03001627'

types = {
    "chair": "03001627",
    "table": "04379243",
    "sofa": "04256520",
    "cabinet": "02933112",
    "lamp": "03636649",
    "car": "02958343",
    "plane": "02691156",
    "watercraft": "04530566"
}

#network details
batch_size = 32
learning_rate = 0.0001
num_epochs = 60


#multiple losses
times_triplet = 100
triplet_margin = 1.0
times_cosine = 100
cosine_margin = 0.5

#saving place
result_path = '../result/base'
text_name_PCN = 'PCN.txt'
text_name = 'result.txt'
model_name_PCN = 'PCN.pt'
model_name_partial = 'partial.pt'
model_name_complete = 'complete.pt'
model_name_decoder = 'decoder.pt'