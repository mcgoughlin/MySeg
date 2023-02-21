import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/ovseg_all_data"
from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2

import gc
import torch
import sys


data_name = str(sys.argv[1])
og_spacing1 = float(sys.argv[2])
og_spacing2 = float(sys.argv[3])
tgt_spacing = float(sys.argv[4])
fold = int(sys.argv[5])

preprocessed_name = '{}mmUHS-{}mmUHP_to_{}mm'.format(og_spacing1,og_spacing2,tgt_spacing)
print(preprocessed_name)
model_name = '6,3x3x3,32_stage2'

vfs = [fold]
lb_classes = [1,2]

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
 
prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                   apply_pooling=False, 
                                   apply_windowing=True,
                                   target_spacing=target_spacing,
                                   lb_classes=lb_classes,
                                   reduce_lb_to_single_class = True,
                                   prev_stage_for_input={'data_name': data_name,
                                                         'preprocessed_name': '{}mm'.format(og_spacing2),
                                                         'model_name': '6,3x3x3,32_ultrahighprec'},
                                   prev_stage_for_mask={'data_name': data_name,
                                                         'preprocessed_name': '{}mm'.format(og_spacing1),
                                                         'model_name': '6,3x3x3,32_ultrahighsens'},
                                   r_dial_mask = 5)

prep.plan_preprocessing_raw_data(data_name)

prep.preprocess_raw_data(raw_data=data_name,
                         preprocessed_name=preprocessed_name)

