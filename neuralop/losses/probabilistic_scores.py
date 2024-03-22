"""
Contains classes of probabilistic metrics and functions for computing scores.
"""

import torch
import torch.nn as nn
import xarray as xr
import xskillscore as xs

class gaussian_crps(object):
    def __init__(self, member_dim=0, reduce_dims=None):
        super().__init__()
        
        self.member_dim = member_dim

        if isinstance(reduce_dims, int):
            self.reduce_dims_list = [reduce_dims]
        else:
            self.reduce_dims_list = reduce_dims
        
        if reduce_dims is not None:
            self.reduce_dims = []
            for dim in self.reduce_dims_list:
                self.reduce_dims.append(f'dim_{dim}')
        else:
            self.reduce_dims = None
    
    def eval(self, x, y):
        x_array = xr.DataArray(x.detach())
        
        # Debug print
        print(x_array.shape)
        
        # Get average and variance over the members of x
        mu = x_array.mean(f'dim_{self.member_dim}')
        sigma = x_array.std(f'dim_{self.member_dim}')
        
        # Make sure y array has correct coordinate labels
        y_array = xr.full_like(mu,0)
        y_array.data = y
        
        # Compute crps with mean over the given dimensions
        crps = xs.crps_gaussian(y_array, mu, sigma, self.reduce_dims)
        
        return crps
    
class cross_entropy(object):
    def __init__(self, member_dim=0):
        super().__init__()
        
        assert member_dim == 0, "Member dim has to be the first one for torchs cross entropy"
        self.loss = nn.CrossEntropyLoss()

    def eval(self, x, y):
        x_un = x.unsqueeze(0)
        y_un = y.unsqueeze(0)
        
        # Depug prints
        print(x_un.shape)
        print(y_un.shape)
        
        # Compute cross entropy
        crs = self.loss(x_un, y_un)
        
        return crs

    
def compute_probabilistic_scores(
    test_db,
    model,
    data_processor,
    losses,
    deterministic=False
):
    """
    Compute scores based on a dictionary of losses ('losses').
    Outputs two dictionaries with the abs and rel scores of all the losses.
    """  
    scores = {}
    for loss_name in losses:
        #print(f'{loss_name}, {losses[loss_name]}')
        scores[loss_name] = 0.0
    
    # Compute score for all losses, averaged over all test samples
    n = test_db.__len__()
    for sample_index in range(n):
        data = test_db[sample_index]
        data = data_processor.preprocess(data, batched=False)
        y = data['y'].squeeze()
        out = model(data['x'].unsqueeze(0))
        # Make sure data is on cpu
        y = y.to(device='cpu')
        out = out.to(device='cpu')
        if deterministic is True:
            out.unsqueeze(0)
        
        for loss_name in losses:
            scores[loss_name] += (1./n)*(losses[loss_name].eval(out, y)).item()
            
    return scores