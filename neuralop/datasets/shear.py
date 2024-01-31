from pathlib import Path
import xarray as xr
import numpy as np
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
    T=20,
):
    """Loads the 2D shear layer dataset

    10.000 samples are available with perturbed initial shear.
    max 8.000 samples (the first ones) are used for training.
    max 2.000 samples (the last ones) are used for validation.

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
    
    return train_loader, test_loaders, data_processor

#------------------------------------------------------------------------------

# The code below was adapted from the CNO repository that uses the same shear data,
# https://github.com/bogdanraonic3/ConvolutionalNeuralOperator/ .

class ShearLayerDataset(Dataset):
    def __init__(
        self,
        res, # fixed, not list
        n,
        channel_dim,
        which, # train, test
        T, # 1,2,3,4
    ):
        """Data location"""
        p = '/cluster/work/math/mroberto-datashare/data_shear/'
        self.file_data = p + f'N{res}_1.nc'
        # XXX 40,000 more available in _0 ... _4
        
        sample_max = 9999 # n_max = 10,000
        sample_train_max = 7999
        
        """Fixed split into train and test datasets"""
        self.length = n
        if which == "training":
            self.start = 10000 # quick fix for missing data
        elif which == "test":
            self.start = sample_train_max  - self.length + 10000
        else:
            print("Flag 'which' of ShearLayerDataset undefined.")

        if res not in [64, 128]:
            raise ValueError(
                f"Only resolutions 64 and 128 are available currently, not {res}."
            )
        self.res = res

        self.ndim = 4 # (batch_size, channels, res, res), see UnitGaussianNormalizer
        # same ndim for x and y
        self.T = T
        
    def __len__(
        self
    ):
        return self.length
        
    def __getitem__(
        self,
        index,
    ):
        var = f'sample_{self.start + index}'

        ds = xr.open_dataset(self.file_data)[var]
        inputs = ds.isel(t=0).to_numpy()
        labels = ds.isel(t=self.T).to_numpy()
        
        ds.close()

        inputs = torch.from_numpy(
            inputs
            # np.expand_dims(inputs,axis=0),
        ).type(torch.float32)

        labels = torch.from_numpy(
            labels
            # np.expand_dims(labels,axis=0)
        ).type(torch.float32)

        return {
            'x':inputs,
            'y':labels,
        }
