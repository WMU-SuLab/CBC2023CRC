import torch
import os

# Check GPU availability
use_cuda = torch.cuda.is_available()
gpu_ids = [0] if use_cuda else []
device = torch.device('cuda' if use_cuda else 'cpu')

#dataset_name = 'ukbb'  # ukbb
dataset_name = 'CBC_ft'  # DRIVE
# dataset_name = 'hrf'  # HRF
# dataset_name = 'all_combine'
dataset = dataset_name
max_step = 30000  # 30000 for ukbb
patch_size_list = [256, 256, 256]
patch_size = patch_size_list[2]
batch_size = 16 # default: 4
print_iter = 100 # default: 100
display_iter = 100 # default: 100
save_iter = 5000 # default: 5000
first_display_metric_iter = max_step-save_iter # default: 25000
lr = 0.00005 # default: 0.0002
step_size = 7000  # 7000 for DRIVE
lr_decay_gamma = 0.5  # default: 0.5
use_SGD = False # default:False

input_nc = 1
ndf = 32
netD_type = 'basic'
n_layers_D = 5
norm = 'instance'
no_lsgan = False
init_type = 'normal'
init_gain = 0.02
use_sigmoid = no_lsgan

# torch.cuda.set_device(gpu_ids[0])
use_GAN = True # default: True

GAN_type = 'vanilla'  # 'vanilla' ,'wgan', 'rank'
treat_fake_cls0 = False
use_noise_input_D = False  # whether use the noisy image as the input of discriminator
use_dropout_D = False  # whether use dropout in each layer of discriminator
vgg_type = 'vgg19'
vgg_layers = [4, 9, 18, 27]
lambda_vgg = 1


# settings for topo loss
use_topo_loss = False  # whether use triplet loss
lambda_topo_list = [1,1,1] # A,V,Vessel
lambda_topo = 0.01

# adam
beta1 = 0.5

# settings for GAN loss
num_classes_D = 1
lambda_GAN_D = 0.01
lambda_GAN_G = 0.01
lambda_GAN_gp = 100
lambda_BCE = 5 if use_GAN else 1
lambda_DICE = 5
lambda_recon = 0
overlap_vessel = 0  # default: 0 (both artery and vein); 1 (artery) ; 2 (vein)

input_nc_D = input_nc + 3

# settings for centerness
use_centerness =False # default: True
dilation_list =  [0] #
lambda_centerness = 1
center_loss_type = 'centerness' # centerness or smoothl1
centerness_map_size =  [128,128]

# pretrained model
use_pretrained_G = True
# model_path_pretrained_G = './log/2023_05_06_09_23_28'
model_path_pretrained_G = './log/2023_07_23_10_28_45'
model_step_pretrained_G = 10000



# path for dataset
stride_height = 50
stride_width = 50


n_classes = 1

model_step = 0


# use CAM
use_CAM = False

#use resize
use_resize = True
reszie_w_h = (256,256)

#use av_cross
use_av_cross = False


# use network
use_network = 'convnext_tiny' # swin_t,convnext_tiny

dataset_path = {'CBC_ft': './data/CBC_ft/test',

                }
trainset_path = dataset_path[dataset_name]


print(f"Dataset: {dataset_name}")
print(f'running at {device}')





