from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import math

import torch
from torch.utils.data import Dataset

from .output_encoder import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding2D
from .data_transforms import DefaultDataProcessor

def load_shear_flow(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    train_resolution=64, 
    test_resolutions=[64, 128],
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=2,
    T=20
):
    """Loads the 2D shear layer dataset

    (10.000 samples are available with perturbed initial shear.
    max 8.000 samples (the first ones) are used for training.
    max 2.000 samples (the last ones) are used for validation.)
    
    For resolution=64 40'000 samples are available for training
    and 10'000 for testing.
    For resolution=128 30'000 amples are available for training 
    and 10'000 for testing.

    Parameters
    ----------
    n_train : int
    n_tests : int
    batch_size : int
    test_batch_sizes : int list
    train_resolution : int, default is 64
    test_resolutions : int list, default is [64,128],
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool, default is True
    encode_input : bool, default is False
    encode_output : bool, default is True
    encoding : 'channel-wise'
    channel_dim : int, default is 1
        where to put the channel dimension, defaults size is 1
        i.e: batch, channel, height, width
    T : index of the time step to predict, one of {1,2,...,20}, default is 20

    Returns
    -------
    training_dataloader, testing_dataloaders

    training_dataloader : torch DataLoader
    testing_dataloaders : dict (key: DataLoader)
    """
    
        
        
    """Load training data"""
    print(
        f"Loading training data at resolution {train_resolution} with {n_train} samples "
        f"and batch-size={batch_size}"
    )
    train_db = ShearLayerDataset(
        train_resolution,
        n_train,
        channel_dim,
        which='training',
        T=T,
    )
        
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    """Load testing data at different resolution"""
    test_loaders = {}
    for (res, n_test, test_batch_size) in zip(
        test_resolutions, n_tests, test_batch_sizes
    ):
        print(
            f"Loading test data at resolution {res} with {n_test} samples "
            f"and batch-size={test_batch_size}"
        )
        test_db = ShearLayerDataset(
            res,
            n_test,
            channel_dim,
            which='test',
            T=T,
        )

        test_loader = torch.utils.data.DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        test_loaders[res] = test_loader 
        
    """Load ensemble data"""
    ensemble_loaders = []
    print(
        f"Loading ensemble data at resolution 128 with 1000 samples "
        f"and batch-size={batch_size}"
    )
    # In distribution
    ensemble_db_inDist = ShearLayerDataset(
        128,
        1000,
        channel_dim,
        which='test',
        T=T,
        ensemble=True,
        outOfDist=False
    )
    
    ensemble_loader_inDist = torch.utils.data.DataLoader(
        ensemble_db_inDist,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    ensemble_loaders.append(ensemble_loader_inDist)

    # Out of distribution
    ensemble_db_outDist = ShearLayerDataset(
        128,
        1000,
        channel_dim,
        which='test',
        T=T,
        ensemble=True,
        outOfDist=True
    )
    
    ensemble_loader_outDist = torch.utils.data.DataLoader(
        ensemble_db_outDist,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    ensemble_loaders.append(ensemble_loader_outDist)
    

    """Input encoder"""
    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(train_db.ndim)) 
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
            reduce_dims = list(range(train_db.ndim))
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
        pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
    else:
        pos_encoding = None
        
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )
    
    return train_loader, test_loaders, ensemble_loaders, data_processor

def plot_shear_flow_test(
    test_db,
    model,
    data_processor,
    n_plot=5,
    save_file='fig-shear.png',
):
    """
    Plots the shear flow dataset.
    Rows: the first n_plot test samples.
    Columns: inputs (t=0), labels (t>0), prediction (t>0).    
    """
    fig = plt.figure(figsize=(7, 2*n_plot))
    for index in range(n_plot):
        
        data = test_db[index]
        data = data_processor.preprocess(data, batched=False)
            
        x = data['x'].unsqueeze(0)
        out = model(x)                  # input u and v

        x = data['x'][0,:,:]            # plot u component only
        y = data['y'].squeeze()[0,:,:]  # plot u component only
        
        # Bring x and y back to cpu memory for the plotting
        x = x.to(device="cpu")
        y = y.to(device="cpu")
        # Same for out
        out = out.to(device="cpu")

        # Transpose to get right direction
        x = torch.transpose(x, 0, 1)
        y = torch.transpose(y, 0, 1)
        out = torch.transpose(out.squeeze()[0,:,:].detach(), 0, 1).numpy()
        
        ax = fig.add_subplot(n_plot, 3, index*3 + 1)
        ax.imshow(x, origin="lower")
        if index == 0: 
            ax.set_title('Input x')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(n_plot, 3, index*3 + 2)
        ax.imshow(y, origin="lower")
        if index == 0: 
            ax.set_title('Ground-truth y')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(n_plot, 3, index*3 + 3)
        ax.imshow(out, origin="lower")
        if index == 0: 
            ax.set_title('Model prediction')
        plt.xticks([], [])
        plt.yticks([], [])
    
    title = 'Inputs, ground-truth output and prediction.'

    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    plt.savefig(save_file)
    fig.show()
    

def plot_shear_flow_test_v(
    test_db,
    model,
    data_processor,
    n_plot=5,
    save_file='fig-shear.png',
):
    """
    Plots the shear flow dataset, v instead of u.
    Rows: the first n_plot test samples.
    Columns: inputs (t=0), labels (t>0), prediction (t>0).    
    """
    fig = plt.figure(figsize=(7, 2*n_plot))
    for index in range(n_plot):
        
        data = test_db[index]
        data = data_processor.preprocess(data, batched=False)
            
        x = data['x'].unsqueeze(0)
        out = model(x)                  # input u and v

        x = data['x'][1,:,:]            # plot v component only
        y = data['y'].squeeze()[1,:,:]  # plot v component only
        
        # Bring x and y back to cpu memory for the plotting
        x = x.to(device="cpu")
        y = y.to(device="cpu")
        # Same for out
        out = out.to(device="cpu")

        # Transpose to get right direction
        x = torch.transpose(x, 0, 1)
        y = torch.transpose(y, 0, 1)
        out = torch.transpose(out.squeeze()[0,:,:].detach(), 0, 1).numpy()
        
        ax = fig.add_subplot(n_plot, 3, index*3 + 1)
        ax.imshow(x, origin="lower")
        if index == 0: 
            ax.set_title('Input x')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(n_plot, 3, index*3 + 2)
        ax.imshow(y, origin="lower")
        if index == 0: 
            ax.set_title('Ground-truth y')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(n_plot, 3, index*3 + 3)
        ax.imshow(out, origin="lower")
        if index == 0: 
            ax.set_title('Model prediction')
        plt.xticks([], [])
        plt.yticks([], [])
    
    title = 'Inputs, ground-truth output and prediction.'

    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    plt.savefig(save_file)
    fig.show()
    
#------------------------------------------------------------------------------

# The code below was adapted from the CNO repository that uses the same shear data,
# https://github.com/bogdanraonic3/ConvolutionalNeuralOperator/ .

class ShearLayerDataset(Dataset):
    """
    Dataset of shear data.
    ensemble==True && outOfDist==True -> stability dataset
    ensemble==True && outOfDist==False -> ensemble dataset
    ensemble==False -> regular shear dataset
    """
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
        
        # Stack u and v
        inputs = np.stack(
            [
                ds['u'].isel(time=0).to_numpy(),
                ds['v'].isel(time=0).to_numpy(),
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

        inputs = torch.from_numpy(
            inputs
        ).type(torch.float32)

        labels = torch.from_numpy(
            labels
        ).type(torch.float32)

        # Return as dictionary
        return {
            'x':inputs,
            'y':labels,
        }
    
    
    def get_ensemble(
        self,
        index
    ):
        """
        UNFINISHED/NEVER USED 
        """
        assert self.ensemble, f'This function can only be called if data set has ensembles.'
        
        if self.ensemble:
            if self.which=='train':
                assert False, f'Macro-Micro currently not available for training.'
            if self.which=='test':
                assert index < 10, f'Requesting ensemble number {index} but only 10 ensembles are available.'
                
        ds = xr.open_zarr(
            self.file_data,
            consolidated=True,
        ).sel(member_macro=index)
        
        return ds

    
    def ensembleInParts(
        self,
        ensemble,
    ):
        """
        Returns lables and inputs of all ensemble members <-> returns each members data at time step T and time step 0
        """
        
        inputs = np.stack(
            [
                ensemble['u'].isel(time=0).to_numpy(),
                ensemble['v'].isel(time=0).to_numpy(),
            ],
            axis=0,
        )
        
        labels = np.stack(
            [
                ensemble['u'].isel(time=self.T).to_numpy(),
                ensemble['v'].isel(time=self.T).to_numpy(),
            ],
            axis=0,
        )
    
        inputs = torch.from_numpy(
            inputs
        ).type(torch.float32)
    
        labels = torch.from_numpy(
            labels
        ).type(torch.float32)
        
        return inputs, labels
