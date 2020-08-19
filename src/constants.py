#place of dataset
scannet_place = '/home/shenguanlin/scannet_extract'
shapenet_place = '/data1/xp/shapenet'
scannet_type_name = 'chair'
shapenet_type_code = '03001627'

#network details
batch_size_shapenet = 8
batch_size_scannet = 8
d_learning_rate = 0.0001
g_learning_rate = 0.0001
decoder_learning_rate = 0.0001

num_epochs_GAN = 10
num_epochs_decoder = 100
d_steps = 1
g_steps = 1