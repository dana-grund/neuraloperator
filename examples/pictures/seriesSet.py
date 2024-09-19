from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import math
#import xskillscore as xs

import torch
from torch.utils.data import Dataset

"""
Shear Layer Dataset that returns a list with all timesteps from 0 to 10 instead just the first and the last.
"""

class ShearLayerSeriesDataset(Dataset):
    def __init__(
        self,
        res, # fixed, not list
        n,
        channel_dim,
        which, # train, test
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
        
        t0 = np.stack(
            [
                ds['u'].isel(time=0).to_numpy(),
                ds['v'].isel(time=0).to_numpy(),
            ],
            axis=0,
        )

        t0 = torch.from_numpy(
            t0
        ).type(torch.float32)

        t1 = np.stack(
            [
                ds['u'].isel(time=1).to_numpy(),
                ds['v'].isel(time=1).to_numpy(),
            ],
            axis=0,
        )

        t1 = torch.from_numpy(
            t1
        ).type(torch.float32)

        t2 = np.stack(
            [
                ds['u'].isel(time=2).to_numpy(),
                ds['v'].isel(time=2).to_numpy(),
            ],
            axis=0,
        )

        t2 = torch.from_numpy(
            t2
        ).type(torch.float32)

        t3 = np.stack(
            [
                ds['u'].isel(time=3).to_numpy(),
                ds['v'].isel(time=3).to_numpy(),
            ],
            axis=0,
        )

        t3 = torch.from_numpy(
            t3
        ).type(torch.float32)

        t4 = np.stack(
            [
                ds['u'].isel(time=4).to_numpy(),
                ds['v'].isel(time=4).to_numpy(),
            ],
            axis=0,
        )

        t4 = torch.from_numpy(
            t4
        ).type(torch.float32)

        t5 = np.stack(
            [
                ds['u'].isel(time=5).to_numpy(),
                ds['v'].isel(time=5).to_numpy(),
            ],
            axis=0,
        )

        t5 = torch.from_numpy(
            t5
        ).type(torch.float32)

        t6 = np.stack(
            [
                ds['u'].isel(time=6).to_numpy(),
                ds['v'].isel(time=6).to_numpy(),
            ],
            axis=0,
        )

        t6 = torch.from_numpy(
            t6
        ).type(torch.float32)

        t7 = np.stack(
            [
                ds['u'].isel(time=7).to_numpy(),
                ds['v'].isel(time=7).to_numpy(),
            ],
            axis=0,
        )

        t7 = torch.from_numpy(
            t7
        ).type(torch.float32)

        t8 = np.stack(
            [
                ds['u'].isel(time=8).to_numpy(),
                ds['v'].isel(time=8).to_numpy(),
            ],
            axis=0,
        )

        t8 = torch.from_numpy(
            t8
        ).type(torch.float32)

        t9 = np.stack(
            [
                ds['u'].isel(time=9).to_numpy(),
                ds['v'].isel(time=9).to_numpy(),
            ],
            axis=0,
        )

        t9 = torch.from_numpy(
            t9
        ).type(torch.float32)

        t10 = np.stack(
            [
                ds['u'].isel(time=10).to_numpy(),
                ds['v'].isel(time=10).to_numpy(),
            ],
            axis=0,
        )
        
        t10 = torch.from_numpy(
            t10
        ).type(torch.float32)

        ds.close()

        return [
            t0,
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            t7,
            t8,
            t9,
            t10
        ]
    
