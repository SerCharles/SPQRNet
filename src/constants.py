#place of dataset
scannet_place = '/home/shenguanlin/scannet_extract'
shapenet_place = '/data1/xp/shapenet'
scannet_type_name = 'chair'
shapenet_type_code = '03001627'

#network details
batch_size = 8
learning_rate = 0.0001

num_epochs_GAN = 100
num_epochs_decoder = 100
d_steps = 1
g_steps = 1

#triplet loss
the_miu = 0
the_lambda = 1