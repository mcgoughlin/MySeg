from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
import argparse
import numpy as np
import os
from ovseg import OV_PREPROCESSED
import os 
import psutil

import threading
import psutil
import pandas as pd
import GPUtil
import time
import contextlib
import gc
import torch

class MonitorUsage(threading.Thread):

    def __init__(self,pid=None):
        super().__init__()
        self.daemon = True
        self.pid = pid
        self.running = True
        
    def run(self):
        self.data = []

        
        while self.running:
            datum = []
            datum.append(psutil.cpu_percent())
            datum.append(psutil.virtual_memory().percent)
            datum.append(GPUtil.getGPUs()[0].memoryUtil*100)
            datum.append(GPUtil.getGPUs()[0].load*100)
            datum.append(time.time())
            self.data.append(datum)
            time.sleep(0.05)
            
        return self.data
    
    def stop(self):
        self.running = False
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False

            
    def __enter__(self):
        self.start()
        # self.run()
# parser = argparse.ArgumentParser()
# parser.add_argument("vf", type=int)
# args = parser.parse_args()
 
data_name = 'kits19_nc'
preprocessed_name = 'casc_stage3'


vfs = [0,1,2,3,4]
lb_classes = [2]

target_spacing=[2.75, 0.8105, 0.8105]

prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                   apply_pooling=False, 
                                   apply_windowing=True,
                                   target_spacing=target_spacing,
                                   lb_classes=lb_classes,
                                   prev_stage_for_input={'data_name': data_name,
                                                         'preprocessed_name': 'casc_stage1',
                                                         'model_name': 'low',
                                                         'data_name': data_name,
                                                         'preprocessed_name': 'casc_stage2',
                                                         'model_name': 'middle'},
                                   prev_stage_for_mask={'data_name': data_name,
                                                         'preprocessed_name': 'casc_stage1',
                                                         'model_name': 'low',
                                                         'data_name': data_name,
                                                         'preprocessed_name': 'casc_stage2',
                                                         'model_name': 'middle'})
prep.plan_preprocessing_raw_data(data_name)

prep.preprocess_raw_data(raw_data=data_name,
                         preprocessed_name=preprocessed_name)

