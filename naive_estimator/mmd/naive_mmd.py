import torch
import time
import torch.nn as nn
import numpy as np
import math
from neuralop.datasets import load_shear_flow
from neuralop.losses import rbf, mmd
from neuralop.models import TFNO
import sys
import psutil
import gc

"""
Functions for computing the MMD for the naive estimator.
"""

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2)} MB")

def naiveBaseline_mmd(dataset, data_processor, set_size):
    #sigma = 0.0313839316368103
    sigma = -0.027283422648906708
    print(f'Gauss kernel sigma for mmd: {sigma}')
    gauss_kernel = rbf(sigma)
    maxMeanDiscr = mmd(gauss_kernel, member_dim=0, reduce_dims=None)

    sqrtSize = math.sqrt(float(set_size))
    if not sqrtSize.is_integer():
        raise Exception("No integer square root.")
    sqrtSize = int(sqrtSize)

    ensemble_x = torch.empty((sqrtSize,2,128,128))
    print(ensemble_x.shape)
    ensemble_y = torch.empty((sqrtSize,2,128,128))

    MMD = 0

    for i in range(sqrtSize):

        for sample_idx in range(sqrtSize):
            data = dataset[i*sqrtSize + sample_idx]
            # data = data_processor.preprocess(data, batched=False)
            #y = data['y'].squeeze()
            #out = model(data['x'].unsqueeze(0))
            # Make sure data is on cpu
            #y = y.to(device='cpu')
            #out = out.to(device='cpu')
            ensemble_x[sample_idx,:,:,:] = data['x'].squeeze().to(device='cpu')
            ensemble_y[sample_idx,:,:,:] = data['y'].squeeze().to(device='cpu')
    
        MMD += (1./float(sqrtSize))*(maxMeanDiscr.eval(ensemble_x, ensemble_y)).item()

        # Clear intermediate results and free up memory
        ensemble_x.detach_()
        ensemble_y.detach_()
        gc.collect()
        print_memory_usage()

    
    return MMD

def naive_mmd(dataset, data_processor, num_ensembles):

    #sigma = 0.0313839316368103
    sigma = -0.027283422648906708
    print(f'Gauss kernel sigma for mmd: {sigma}')
    gauss_kernel = rbf(sigma)
    maxMeanDiscr = mmd(gauss_kernel, member_dim=0, reduce_dims=None)

    ensemble_x = torch.empty((100,2,128,128))
    print(ensemble_x.shape)
    ensemble_y = torch.empty((100,2,128,128))

    MMD = 0

    for i in range(num_ensembles):

        for sample_idx in range(100):
            data = dataset[i*100 + sample_idx]
            ensemble_x[sample_idx,:,:,:] = data['x'].squeeze().to(device='cpu')
            ensemble_y[sample_idx,:,:,:] = data['y'].squeeze().to(device='cpu')
        
        # for i in range(100):
        #     NO_crps += (1./float(num_ensembles*100))*np.mean(sr.crps_ensemble(ensemble_x.detach(), ensemble_y[i,:,:,:].detach(), axis=0), None)
        #     Num_crps += (1./float(num_ensembles*100))*np.mean(sr.crps_ensemble(ensemble_y.detach(), ensemble_y[i,:,:,:].detach(), axis=0), None)
    
        MMD += (1./float(num_ensembles))*(maxMeanDiscr.eval(ensemble_x, ensemble_y)).item()

        # Clear intermediate results and free up memory
        ensemble_x.detach_()
        ensemble_y.detach_()
        gc.collect()
        print_memory_usage()

    return MMD





