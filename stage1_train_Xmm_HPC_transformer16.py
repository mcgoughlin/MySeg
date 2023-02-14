import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/ovseg_all_data"

from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

import gc
import torch
import sys


data_name = str(sys.argv[1])
spacing = float(sys.argv[2])
fold = int(sys.argv[3])
hd_weight = float(sys.argv[4])

preprocessed_name = '{}mm_distmaps'.format(spacing)
model_name = '5,3,32_distmaps_hdw{}_transformer'.format(hd_weight)

vfs = [fold]
lb_classes = [1]

target_spacing=[10,10,10]
kits19ncct_dataset_properties ={
    		'median_shape' : [104., 512., 512.],
    		'median_spacing' : [2.75,      0.8105, 0.8105],
    		'fg_percentiles' : [-118, 136 ],
    		'percentiles' : [0.5, 99.5],
    		'scaling_foreground' : [ 42.94977,  11.57459],
    		'n_fg_classes' : 1,
    		'scaling_global' : [ 510.60403, -431.1344],
    		'scaling_window' : [ 68.93295,  -65.79061]}


patch_size = [32,32,32]
#patch dimension must be divisible by respective (((kernel_dimension+1)//2)^depth)/2
#Patch size dictates input size to CNN: input dim (metres) = patch_size*target_spacing/1000
#finally, depth and conv kernel size dictate attentive area - importantly different to input size:
# attentive_area (in each dimension, metres) = input size / bottom encoder spatial dim
#                                           = ((((kernel_dimension+1)//2)^depth)/2)*target_spacing/1000
z_to_xy_ratio = 1
larger_res_encoder = True
n_fg_classes = 1


model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     n_fg_classes=n_fg_classes,
                                                     use_prg_trn=False)


model_params['architecture'] = 'UNetTransformer'
model_params['network']['kernel_sizes'] =5*[(3,3,3)]
model_params['network']['norm'] = 'inst'
model_params['network']['in_channels']=1
model_params['network']['filters']=32
model_params['network']['filters_max']=2048
del model_params['network']['block']
del model_params['network']['z_to_xy_ratio']
del model_params['network']['n_blocks_list']
del model_params['network']['stochdepth_rate']

lr=0.0001
# dist_flag settings
model_params['training']['dist_flag'] = True
model_params['training']['hd_weight'] = hd_weight
# model_params['data']['folders'] = ['images', 'labels']#, 'masks']
# model_params['data']['keys'] = ['image', 'label']#, 'mask']
model_params['data']['folders'] = ['images', 'labels','dist_maps']#, 'masks']
model_params['data']['keys'] = ['image', 'label','dist_map']#, 'mask']

model_params['training']['num_epochs'] = 250
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr': lr,
                                            'betas': (0.95, 0.9),
                                            'eps': 1e-08}
model_params['training']['lr_params'] = {'n_warmup_epochs': 5, 'lr_max': 0.005}
model_params['data']['trn_dl_params']['epoch_len']=500
model_params['data']['trn_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
model_params['data']['val_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
model_params['training']['lr_schedule'] = 'lin_ascent_log_decay'
model_params['training']['lr_exponent'] = 4
model_params['training']['loss_params'] = {'loss_names':['cross_entropy','dice_loss','sensitivity_loss'], 
                                           'loss_weights':[0.2,0.2,0.6]}
model_params['data']['trn_dl_params']['batch_size']=16
model_params['data']['val_dl_params']['epoch_len']=50
# model_params['postprocessing'] = {'mask_with_reg': True}

for vf in vfs:
    model = SegmentationModel(val_fold=vf,
                                data_name=data_name,
                                preprocessed_name=preprocessed_name,
                                model_name=model_name,
                                model_parameters=model_params)
    torch.cuda.empty_cache()
    gc.collect()

    model.training.train()
    torch.cuda.empty_cache()
    gc.collect()
    
    model.eval_validation_set()
    torch.cuda.empty_cache()
    gc.collect()
