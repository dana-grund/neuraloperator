import torch
import time
import torch.nn as nn
import scoringrules as sr
import numpy as np
import math
from neuralop.datasets import load_shear_flow
#from neuralop.losses import 
from neuralop.models import TFNO
import sys
import psutil
import gc

"""
Functions for computing the naive CRPS.
"""

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2)} MB")

def naiveBaseline_crps(dataset, data_processor, set_size):

    sqrtSize = math.sqrt(float(set_size))
    if not sqrtSize.is_integer():
        raise Exception("No integer square root.")
    sqrtSize = int(sqrtSize)
    ensemble_x = torch.empty((sqrtSize,2,128,128))
    print(ensemble_x.shape)
    ensemble_y = torch.empty((sqrtSize,2,128,128))

    NO_crps = 0.0
    Num_crps = 0.0

    for i in range(sqrtSize):

        for sample_idx in range(sqrtSize):
            data = dataset[i*sqrtSize + sample_idx]
            #data = data_processor.preprocess(data, batched=False)
            ensemble_x[sample_idx,:,:,:] = data['x'].squeeze().to(device='cpu')
            ensemble_y[sample_idx,:,:,:] = data['y'].squeeze().to(device='cpu')
        
        for i in range(sqrtSize):
            NO_crps += (1./float(set_size))*np.mean(sr.crps_ensemble(ensemble_x.detach(), ensemble_y[i,:,:,:].detach(), axis=0), None)
            Num_crps += (1./float(set_size))*np.mean(sr.crps_ensemble(ensemble_y.detach(), ensemble_y[i,:,:,:].detach(), axis=0), None)
    
        # Clear intermediate results and free up memory
        ensemble_x.detach_()
        ensemble_y.detach_()
        gc.collect()
        print_memory_usage()

    return abs(NO_crps - Num_crps)


def naive_crps(dataset, data_processor, num_ensembles):

    ensemble_x = torch.empty((100,2,128,128))
    print(ensemble_x.shape)
    ensemble_y = torch.empty((100,2,128,128))

    NO_crps = 0.0
    Num_crps = 0.0

    for i in range(num_ensembles):

        for sample_idx in range(100):
            data = dataset[i*100 + sample_idx]
            ensemble_x[sample_idx,:,:,:] = data['x'].squeeze().to(device='cpu')
            ensemble_y[sample_idx,:,:,:] = data['y'].squeeze().to(device='cpu')
        
        for i in range(100):
            NO_crps += (1./float(num_ensembles*100))*np.mean(sr.crps_ensemble(ensemble_x.detach(), ensemble_y[i,:,:,:].detach(), axis=0), None)
            Num_crps += (1./float(num_ensembles*100))*np.mean(sr.crps_ensemble(ensemble_y.detach(), ensemble_y[i,:,:,:].detach(), axis=0), None)
    
        # Clear intermediate results and free up memory
        ensemble_x.detach_()
        ensemble_y.detach_()
        gc.collect()
        print_memory_usage()

    return abs(NO_crps - Num_crps)





