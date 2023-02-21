#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:36:08 2023

@author: mcgoug01
"""
import os
import nibabel as nib
import matplotlib.pyplot as plt
import gc
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as distance


def generate_dtm(outputs_soft,pred=True):
    seg_dtm_npy = _calculate_distance(outputs_soft<0.5)
    return seg_dtm_npy
    
def _calculate_distance(outputs_sft):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """
    out_shape = outputs_sft.shape
    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = outputs_sft[b].astype(bool)
        if posmask.any():
            posdis = distance(posmask)
            fg_dtm[b] = posdis

    return fg_dtm
    
    
    