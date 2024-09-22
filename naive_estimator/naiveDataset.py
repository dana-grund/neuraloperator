from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import math
#import xskillscore as xs

import torch
from torch.utils.data import Dataset
import sys
from local_output_encoder import UnitGaussianNormalizer

sys.path.insert(len(sys.path), '/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/neuralop/datasets')

from tensor_dataset import TensorDataset
from transforms import PositionalEmbedding2D
from data_transforms import DefaultDataProcessor

"""
Dataset that serves as a naive estimator by outputing T and T-1 values instead of T and 0 values like the regular dataset.
Includes functions for acquire different types of naive estimators.
"""

class ShearLayerNaiveDataset(Dataset):
    def __init__(
        self,
        res, # fixed, not list
        n,
        channel_dim,
        which, # train, test
        T, # 1,2,3,4
        ensemble=False,
        outOfDist=False
    ):
        """Data location"""
        p = '/cluster/work/climate/webesimo'
        if ensemble:
            p = '/cluster/work/climate/dgrund/data_shear_layer_2D_macro_micro/'
            if outOfDist:
                self.file_data = os.path.join(
                    p, 'macro_micro_2d_id_2_clean.zarr'
                )
            else:
                self.file_data = os.path.join(
                    p, 'macro_micro_2d_id_3_corrected_clean.zarr'
                )
        else:
            self.file_data = os.path.join(
                p, f'data_N{res}.zarr'
            )
        
        if res not in [64, 128]:
            raise ValueError(
                f"Only resolutions 64 and 128 are available currently, not {res}."
            )
        self.res = res
            
        """Fixed split into train and test datasets"""
        self.length = n
        if ensemble:
            if which == "training":
                assert False, f'Ensemble data currently not available for training' #self.start = 1000
            elif which == "test":
                self.start = 0
            else:
                print("Flag 'which' of ShearLayerDataset undefined.")
        else:
            if which == "training": # 40,000 samples for both 64 and 128 resolution
                self.start = 0
            elif which == "test": # 10,000 samples for both
                self.start = 40000
            else:
                print("Flag 'which' of ShearLayerDataset undefined.")

        self.which = which
        self.ndim = 4 # (batch_size, channels, res, res), see UnitGaussianNormalizer
        # same ndim for x and y
        if T == 0:
            assert False, "No naive estimate available for T=0."
        self.T = T
        self.ensemble = ensemble
        
    def __len__(
        self
    ):
        return self.length
        
    def __getitem__(
        self,
        index,
    ):
        assert index >= 0, 'Only positive indexing'
        if self.ensemble:
            if self.which=='train':
                assert False, f'Macro-Micro currently not available for training.'
            if self.which=='test':
                assert index < 1000, f'Requesting index {index} for testing but only 1000 are available.'
        else:
            if self.which=='train':
                assert index < 40000, f'Requesting index {index} for training but only 40_000 are available.'
        
            if self.which=='test':
                assert index < 10000, f'Requesting index {index} for testing but only 10_000 are available.'
        
        macro = self.start + index
        micro = 0
        if self.ensemble:
            # Different indexation for ensemble dataset. 100 micros per macro
            macro = math.floor(index/100)
            micro = index % 100
        
        ds = xr.open_zarr(
            self.file_data,
            consolidated=True,
        ).sel(member_macro=macro).sel(member_micro=micro)
        
        estimates = np.stack(
            [
                ds['u'].isel(time=self.T-1).to_numpy(),
                ds['v'].isel(time=self.T-1).to_numpy(),
            ],
            axis=0,
        )
        labels = np.stack(
            [
                ds['u'].isel(time=self.T).to_numpy(),
                ds['v'].isel(time=self.T).to_numpy(),
            ],
            axis=0,
        )
        
        ds.close()

        estimates = torch.from_numpy(
            estimates
        ).type(torch.float32)

        labels = torch.from_numpy(
            labels
        ).type(torch.float32)

        return {
            'x':estimates,
            'y':labels,
        }
    
def get_ensembleSet():
    ensemble_db = ShearLayerNaiveDataset(
        128,
        1000,
        2,
        which='test',
        T=10,
        ensemble=True,
        outOfDist=False
    )
    return ensemble_db

def get_stabilitySet():
    stability_db = ShearLayerNaiveDataset(
        128,
        1000,
        2,
        which='test',
        T=10,
        ensemble=True,
        outOfDist=True
    )
    return stability_db

def get_baselineSet(set_size):
    test_db = ShearLayerNaiveDataset(
        128,
        set_size,
        2,
        which='test',
        T=10,
    )
    return test_db

def get_processor(some_db, encode_input=False, encode_output=True, positional_encoding=True, encoding="channel-wise"):
    """Input encoder"""
    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(some_db.ndim)) 
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        # output_encoder.fit(y_train) # to avoid loading all data at once, fill in by hand
        # XXX check if normalization works as expected
        
        input_encoder.n_elements = None 
        input_encoder.mean = torch.tensor(
            [0, 0], 
            dtype=torch.float32
        ).unsqueeze(1).unsqueeze(1)
        input_encoder.squared_mean = input_encoder.mean
        input_encoder.std = torch.tensor(
            [0.49198706571149564, 0.36194905497513363], 
            dtype=torch.float32
        ).unsqueeze(1).unsqueeze(1)

    else:
        input_encoder = None
        
    """Output encoder"""
    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(some_db.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        # output_encoder.fit(y_train)
        
        output_encoder.n_elements = None
        output_encoder.mean = torch.tensor(
            [0, 0], 
            dtype=torch.float32
        ).unsqueeze(1).unsqueeze(1)
        output_encoder.squared_mean = output_encoder.mean
        output_encoder.std = torch.tensor(
            [0.49198706571149564, 0.36194905497513363], 
            dtype=torch.float32
        ).unsqueeze(1).unsqueeze(1)
    else:
        output_encoder = None

    """Positional encoding"""
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries=[[0, 1], [0, 1]])
    else:
        pos_encoding = None
        
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )

    return data_processor