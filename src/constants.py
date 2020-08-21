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
times_reconstruction = 30

#saving place
result_path = '../result/result.txt'
model_path_partial = '../result/partial.pt'
model_path_complete = '../result/complete.pt'
model_path_decoder = '../result/decoder.pt'