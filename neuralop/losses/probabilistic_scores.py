"""
Contains classes of probabilistic metrics and functions for computing scores.
"""

import torch
import torch.nn as nn
import xarray as xr
import xskillscore as xs
import scoringrules as sr
import numpy as np

class gaussian_crps(object):
    """
    Class for computing gaussian crps. Takes average over ground truth ensemble.
    """
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
        """
        X and Y both need to be ensembles.
        """
        x_array = xr.DataArray(x.detach())
        y_array = xr.DataArray(y.detach())
        
        # Get average and variance over the members of x
        mu = x_array.mean(f'dim_{self.member_dim}')
        sigma = x_array.std(f'dim_{self.member_dim}')
        
        # Average over Ys ensemble members
        y_avgs = y_array.mean(f'dim_{self.member_dim}')
        
        # Compute crps with mean over the given dimensions
        crps = xs.crps_gaussian(y_avgs, mu, sigma, self.reduce_dims)
        
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
    
class hacky_crps(object):
    """
    Class for computing hacky workaround of gaussian crps.
    """
    def __init__(self, member_dim=0, reduce_dims=None):
        super().__init__()
        
        self.member_dim = member_dim

        if isinstance(reduce_dims, list):
            self.reduce_dims = tuple(reduce_dims)
        else:
            self.reduce_dims = reduce_dims
            
    def eval(self, ensemble_x, ensemble_y):
        """
        X and Y both need to be ensembles.
        """
        NO_crps = 0.0
        Num_crps = 0.0
        
        x_mu = torch.mean(ensemble_x, self.member_dim).detach()
        y_mu = torch.mean(ensemble_y, self.member_dim).detach()
        
        x_sigma = torch.std(ensemble_x, dim=self.member_dim).detach()
        y_sigma = torch.std(ensemble_y, dim=self.member_dim).detach()
        
        if self.member_dim == 0:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_normal(x_mu, x_sigma, ensemble_y[i,:,:,:].detach()), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_normal(y_mu, y_sigma, ensemble_y[i,:,:,:].detach()), self.reduce_dims)
        elif self.member_dim == 1:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_normal(x_mu, x_sigma, ensemble_y[:,i,:,:].detach()), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_normal(y_mu, y_sigma, ensemble_y[:,i,:,:].detach()), self.reduce_dims)
        elif self.member_dim == 2:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_normal(x_mu, x_sigma, ensemble_y[:,:,i,:].detach()), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_normal(y_mu, y_sigma, ensemble_y[:,:,i,:].detach()), self.reduce_dims)
        else:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_normal(x_mu, x_sigma, ensemble_y[:,:,:,i].detach()), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_normal(y_mu, y_sigma, ensemble_y[:,:,:,i].detach()), self.reduce_dims)
        
        return abs(NO_crps - Num_crps)

    
def compute_probabilistic_scores(
    test_db,
    model,
    data_processor,
    losses,
    deterministic=False
):
    """
    Compute scores based on a dictionary of losses ('losses').
    Outputs two dictionaries with the scores of all the losses.
    """  
    scores = {}
    for loss_name in losses:
        #print(f'{loss_name}, {losses[loss_name]}')
        scores[loss_name] = 0.0
    
    # Compute score for all losses, averaged over all ensembles # Ensemble wise approach
    #for index in range(10):
    #    ensemble = test_db.get_ensemble(index)
        
        # Get truth and inputs
    #    inputs, truth = test_db.ensembleInParts(ensemble)
        
        # Predict
    #    predictions = evaluate_ensemble(model, inputs)
        
        #data = data_processor.preprocess(data, batched=False)
        #y = data['y'].squeeze()
        #out = model(data['x'].unsqueeze(0))
        # Make sure data is on cpu
        #y = y.to(device='cpu')
        #out = out.to(device='cpu')
        #if deterministic is True:
        #    out.unsqueeze(0)
        
    #    for loss_name in losses:
            # Average over all ensembles
    #        scores[loss_name] += (1./n)*(losses[loss_name].eval(predictions, truth)).item()
    
    # "Normla approach"
    hacky_crps_average = 0.0
    for ens_idx in range(10):
        
        ensemble_x = torch.empty((100,2,128,128))
        ensemble_y = torch.empty((100,2,128,128))
        
        for sample_idx in range(100):
            data = test_db[ens_idx*100 + sample_idx]
            data = data_processor.preprocess(data, batched=False)
            #y = data['y'].squeeze()
            #out = model(data['x'].unsqueeze(0))
            # Make sure data is on cpu
            #y = y.to(device='cpu')
            #out = out.to(device='cpu')
            ensemble_x[sample_idx,:,:,:] = model(data['x'].unsqueeze(0)).to(device='cpu')
            ensemble_y[sample_idx,:,:,:] = data['y'].squeeze().to(device='cpu')
            
        NO_crps, Num_crps = hacky_crps_comp(ensemble_x, ensemble_y)
        hacky_crps_average += (1./10)*(Num_crps - NO_crps)
            
        for loss_name in losses:
            scores[loss_name] += (1./10)*(losses[loss_name].eval(ensemble_x, ensemble_y)).item()
            
    print(f"\nHacky crps average: {abs(hacky_crps_average)}")
    return scores

def evaluate_ensemble(
    model,
    inputs,
):
    """
    Takes ensemble of input values and returns predicitons for all micro members.
    Arguments:
        model: the model to evaluate the data with
        inputs: array of ensemble at timestep 0
    Retuerns:
        predictions: tensor of shape=(100, 2, 128, 128)
    """
    predictions = np.empty((100, 2, 128, 128))
    # Loop over all members
    for i in range(100):
        predictions[i,:,:,:] = model(inputs[i,:,:,:])
    
    
def hacky_crps_comp(
    ensemble_x,
    ensemble_y
):
    """
    """
    NO_crps = 0.0
    Num_crps = 0.0
    
    x_mu = torch.mean(ensemble_x, 0).detach()
    y_mu = torch.mean(ensemble_y, 0).detach()
    
    x_sigma = torch.std(ensemble_x, dim=0).detach()
    y_sigma = torch.std(ensemble_y, dim=0).detach()
    
    for i in range(100):
        NO_crps += (1./100.)*np.mean(sr.crps_normal(x_mu, x_sigma, ensemble_y[i,:,:,:].detach()))
        Num_crps += (1./100.)*np.mean(sr.crps_normal(y_mu, y_sigma, ensemble_y[i,:,:,:].detach()))
        
    return NO_crps, Num_crps