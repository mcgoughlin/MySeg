import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/ovseg_all_data"
from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2

import gc
import torch
import sys


data_name = str(sys.argv[1])
og_spacing = float(sys.argv[2])
tgt_spacing = float(sys.argv[3])
fold = int(sys.argv[4])

preprocessed_name = '{}mm_ultrahighsens_to_{}mm'.format(og_spacing,tgt_spacing)
print(preprocessed_name)
model_name = '6,3x3x3,32_stage2'

vfs = [fold]
lb_classes = [1]

target_spacing=[tgt_spacing]*3

kits19ncct_dataset_properties ={
    		'median_shape' : [104., 512., 512.],
    		'median_spacing' : [2.75,      0.8105, 0.8105],
    		'fg_percentiles' : [-118, 136 ],
    		'percentiles' : [0.5, 99.5],
    		'scaling_foreground' : [ 42.94977,  11.57459],
    		'n_fg_classes' : 1,
    		'scaling_global' : [ 510.60403, -431.1344],
    		'scaling_window' : [ 68.93295,  -65.79061]}
 

vfs = [0,1,2,3,4]
lb_classes = [1]

prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                   apply_pooling=False, 
                                   apply_windowing=True,
                                   target_spacing=target_spacing,
                                   lb_classes=lb_classes,
                                   prev_stage_for_input={'data_name': data_name,
                                                         'preprocessed_name': '{}mm'.format(og_spacing),
                                                         'model_name': '5,3x3x3,32_ultrahighsens'},
                                   prev_stage_for_mask={'data_name': data_name,
                                                         'preprocessed_name': '{}mm'.format(og_spacing),
                                                         'model_name': '5,3x3x3,32_ultrahighsens'})
prep.plan_preprocessing_raw_data(data_name)

prep.preprocess_raw_data(raw_data=data_name,
                         preprocessed_name=preprocessed_name)

