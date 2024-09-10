"""
Contains classes of probabilistic metrics and functions for computing scores.
"""

import torch
import torch.nn as nn
import xarray as xr
import xskillscore as xs
import scoringrules as sr
import numpy as np
import math
import sys
import psutil
import gc

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
        
        # Compute means
        x_mu = torch.mean(ensemble_x, self.member_dim).detach()
        y_mu = torch.mean(ensemble_y, self.member_dim).detach()
        
        # Compute stds
        x_sigma = torch.std(ensemble_x, dim=self.member_dim).detach()
        y_sigma = torch.std(ensemble_y, dim=self.member_dim).detach()
        
        # Compute CRPS for predicted and ground truth ensemble. Indexing dependent on member dim
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
        
        # Take difference of ground truth and prediction crps
        return abs(NO_crps - Num_crps)

class lognormal_crps(object):
    """
    Class for computing lognormal crps. Follows 'hacky' approach.
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
        
        x_data = ensemble_x.detach().numpy().copy()
        y_data = ensemble_y.detach().numpy().copy()
        
        # Mean and standard dev. for standardization
        x_mu_old = np.mean(x_data, axis=self.member_dim)
        x_sigma_old = np.std(x_data, axis=self.member_dim)
        y_mu_old = np.mean(y_data, axis=self.member_dim)
        y_sigma_old = np.std(y_data, axis=self.member_dim)
        
        # Standardize
        if self.member_dim == 0:
            for i in range(100):
                x_data[i,:,:,:] = (x_data[i,:,:,:] - x_mu_old) / x_sigma_old
                y_data[i,:,:,:] = (y_data[i,:,:,:] - y_mu_old) / y_sigma_old
        elif self.member_dim == 1:
            for i in range(100):
                x_data[:,i,:,:] = (x_data[:,i,:,:] - x_mu_old) / x_sigma_old
                y_data[:,i,:,:] = (y_data[:,i,:,:] - y_mu_old) / y_sigma_old
        elif self.member_dim == 2:
            for i in range(100):
                x_data[:,:,i,:] = (x_data[:,:,i,:] - x_mu_old) / x_sigma_old
                y_data[:,:,i,:] = (y_data[:,:,i,:] - y_mu_old) / y_sigma_old
        else:
            for i in range(100):
                x_data[:,:,:,i] = (x_data[:,:,:,i] - x_mu_old) / x_sigma_old
                y_data[:,:,:,i] = (y_data[:,:,:,i] - y_mu_old) / y_sigma_old
                
        # Min for shifting to positive
        x_mins = np.min(x_data, axis=self.member_dim)
        y_mins = np.min(y_data, axis=self.member_dim)
        
        # Shift (+ 1 to avoid dividing by zero)
        if self.member_dim == 0:
            for i in range(100):
                x_data[i,:,:,:] += np.abs(x_mins) + 1.0
                y_data[i,:,:,:] += np.abs(y_mins) + 1.0
        elif self.member_dim == 1:
            for i in range(100):
                x_data[:,i,:,:] += np.abs(x_mins) + 1.0
                y_data[:,i,:,:] += np.abs(y_mins) + 1.0
        elif self.member_dim == 2:
            for i in range(100):
                x_data[:,:,i,:] += np.abs(x_mins) + 1.0
                y_data[:,:,i,:] += np.abs(y_mins) + 1.0
        else:
            for i in range(100):
                x_data[:,:,:,i] += np.abs(x_mins) + 1.0
                y_data[:,:,:,i] += np.abs(y_mins) + 1.0
        
        # Compute new means and stds
        x_mu = np.mean(x_data, axis=self.member_dim)
        y_mu = np.mean(y_data, axis=self.member_dim)
        x_sigma = np.std(x_data, axis=self.member_dim)
        y_sigma = np.std(y_data, axis=self.member_dim)
        
        # Compute CRPS for predicted and ground truth ensemble. Indexing dependent on member dim
        if self.member_dim == 0:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_lognormal(x_mu, x_sigma, y_data[i,:,:,:]), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_lognormal(y_mu, y_sigma, y_data[i,:,:,:]), self.reduce_dims)
        elif self.member_dim == 1:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_lognormal(x_mu, x_sigma, y_data[:,i,:,:]), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_lognormal(y_mu, y_sigma, y_data[:,i,:,:]), self.reduce_dims)
        elif self.member_dim == 2:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_lognormal(x_mu, x_sigma, y_data[:,:,i,:]), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_lognormal(y_mu, y_sigma, y_data[:,:,i,:]), self.reduce_dims)
        else:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_lognormal(x_mu, x_sigma, y_data[:,:,:,i]), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_lognormal(y_mu, y_sigma, y_data[:,:,:,i]), self.reduce_dims)
        
        # Take difference of ground truth and prediction crps
        return abs(NO_crps - Num_crps)


class ensemble_crps(object):
    """
    Class for computing ensemble crps. Follows 'hacky' approach
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
        
        # Compute CRPS for predicted and ground truth ensemble. Indexing dependent on member dim
        if self.member_dim == 0:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_ensemble(ensemble_x.detach(), ensemble_y[i,:,:,:].detach(), axis=0), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_ensemble(ensemble_y.detach(), ensemble_y[i,:,:,:].detach(), axis=0), self.reduce_dims)
        elif self.member_dim == 1:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_ensemble(ensemble_x.detach(), ensemble_y[:,i,:,:].detach(), axis=1), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_ensemble(ensemble_y.detach(), ensemble_y[:,i,:,:].detach(), axis=1), self.reduce_dims)
        elif self.member_dim == 2:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_ensemble(ensemble_x.detach(), ensemble_y[:,:,i,:].detach(), axis=2), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_ensemble(ensemble_y.detach(), ensemble_y[:,:,i,:].detach(), axis=2), self.reduce_dims)
        else:
            for i in range(100):
                NO_crps += (1./100.)*np.mean(sr.crps_ensemble(ensemble_x.detach(), ensemble_y[:,:,:,i].detach(), axis=3), self.reduce_dims)
                Num_crps += (1./100.)*np.mean(sr.crps_ensemble(ensemble_y.detach(), ensemble_y[:,:,:,i].detach(), axis=3), self.reduce_dims)
        
        # Take difference of ground truth and prediction crps
        return abs(NO_crps - Num_crps)
    

# Maximum mean discrepancy
class mmd(object):
    """
    Class for computing maximum mean discrepancy.
    """
    def __init__(self, kernel, member_dim=0, reduce_dims=None):
        super().__init__()
        
        self.member_dim = member_dim

        if isinstance(reduce_dims, list):
            self.reduce_dims = tuple(reduce_dims)
        else:
            self.reduce_dims = reduce_dims
          
        self.kernel = kernel
            
    def eval(self, ensemble_x, ensemble_y):
        """
        X and Y both need to be ensembles. Computation according to formula here: https://www.youtube.com/watch?v=zFffYuDGslg (7:50)
        """
        xx_mean = 0.0
        yy_mean = 0.0
        xy_mean = 0.0
        ensemble_size = ensemble_x.shape[self.member_dim]
        count = 0

        # Compute different means neccessary for computing mmd. Indexing dependent on member dim
        if self.member_dim == 0:
            for i in range(ensemble_size):
                for j in range(i, ensemble_size):
                    xx_mean += torch.mean(self.kernel(ensemble_x[i,:,:,:], ensemble_x[j,:,:,:]), self.reduce_dims)
                    yy_mean += torch.mean(self.kernel(ensemble_y[i,:,:,:], ensemble_y[j,:,:,:]), self.reduce_dims)
                    xy_mean += torch.mean(self.kernel(ensemble_x[i,:,:,:], ensemble_y[j,:,:,:]), self.reduce_dims)
                    count += 1
        elif self.member_dim == 1:
            for i in range(ensemble_size):
                for j in range(i, ensemble_size):
                    xx_mean += torch.mean(self.kernel(ensemble_x[:,i,:,:], ensemble_x[:,j,:,:]), self.reduce_dims)
                    yy_mean += torch.mean(self.kernel(ensemble_y[:,i,:,:], ensemble_y[:,j,:,:]), self.reduce_dims)
                    xy_mean += torch.mean(self.kernel(ensemble_x[:,i,:,:], ensemble_y[:,j,:,:]), self.reduce_dims)
                    count += 1
        elif self.member_dim == 2:
            for i in range(ensemble_size):
                for j in range(i, ensemble_size):
                    xx_mean += torch.mean(self.kernel(ensemble_x[:,:,i,:], ensemble_x[:,:,j,:]), self.reduce_dims)
                    yy_mean += torch.mean(self.kernel(ensemble_y[:,:,i,:], ensemble_y[:,:,j,:]), self.reduce_dims)
                    xy_mean += torch.mean(self.kernel(ensemble_x[:,:,i,:], ensemble_y[:,:,j,:]), self.reduce_dims)
                    count += 1
        else:
            for i in range(ensemble_size):
                for j in range(i, ensemble_size):
                    xx_mean += torch.mean(self.kernel(ensemble_x[:,:,:,i], ensemble_x[:,:,:,j]), self.reduce_dims)
                    yy_mean += torch.mean(self.kernel(ensemble_y[:,:,:,i], ensemble_y[:,:,:,j]), self.reduce_dims)
                    xy_mean += torch.mean(self.kernel(ensemble_x[:,:,:,i], ensemble_y[:,:,:,j]), self.reduce_dims)
                    count += 1
        xx_mean /= count
        yy_mean /= count
        xy_mean /= count

        return torch.as_tensor(math.sqrt(xx_mean + yy_mean - 2*xy_mean))
        
    
# Kernels for mmd
class rbf(object):
    """
    Gaussian kernel for mmd. Formula: exp(-(||x-y||²) / sigma²)
    """
    def __init__(self, sigma):
        super().__init__()
        
        self.sigma_sq = sigma*sigma
        
    
    def __call__(self, x, y):
        """
        x and y are tensors.
        """
        return torch.exp(- torch.square(y - x) / (2*self.sigma_sq))

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
        scores[loss_name] = 0.0
    
    # Loop over all ensembles
    for ens_idx in range(10):
        
        ensemble_x = torch.empty((100,2,128,128))
        ensemble_y = torch.empty((100,2,128,128))
        
        # Get predictions and ground truths for all 100 members
        for sample_idx in range(100):
            data = test_db[ens_idx*100 + sample_idx]
            data = data_processor.preprocess(data, batched=False)
            ensemble_x[sample_idx,:,:,:] = model(data['x'].unsqueeze(0)).to(device='cpu')
            ensemble_y[sample_idx,:,:,:] = data['y'].squeeze().to(device='cpu')
            
        # Compute all losses for this ensemble
        for loss_name in losses:
            scores[loss_name] += (1./10)*(losses[loss_name].eval(ensemble_x, ensemble_y)).item()
            
    return scores

def evaluate_ensemble(
    model,
    inputs,
):
    """
    UNFINISHED
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
    NEVER USED
    Function for computing gaussian crps for an ensemble.
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


def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2)} MB")


def baseline_crps(model, dataset, data_processor, set_size):
    """
    Function for computing the baseline crps. dataset is expected to be the regular shear data (baseline dataset).
    """

    model.eval()

    sqrtSize = math.sqrt(float(set_size))
    if not sqrtSize.is_integer():
        raise Exception("No integer square root.")
    sqrtSize = int(sqrtSize)
    ensemble_x = torch.empty((sqrtSize,2,128,128))
    ensemble_y = torch.empty((sqrtSize,2,128,128))

    NO_crps = 0.0
    Num_crps = 0.0

    # Compute as average of sqrtSize number of "artificial" ensembles instead of one huge ensemble
    # (in order to use less memory)
    for i in range(sqrtSize):

        for sample_idx in range(sqrtSize):
            data = dataset[i*sqrtSize + sample_idx]
            data = data_processor.preprocess(data, batched=False)
            out = model(data['x'].unsqueeze(0)).to(device='cpu')
            out, data = data_processor.postprocess(out, data)
            ensemble_x[sample_idx,:,:,:] = out
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


def baseline_mmd(model, dataset, data_processor, set_size):
    """
    Function for computing the baseline mmd. dataset is expected to be the regular shear data (baseline dataset).
    """
    #sigma = 0.0313839316368103
    sigma = -0.027283422648906708
    print(f'Gauss kernel sigma for mmd: {sigma}')
    gauss_kernel = rbf(sigma)
    maxMeanDiscr = mmd(gauss_kernel, member_dim=0, reduce_dims=None)

    model.eval()

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
            data = data_processor.preprocess(data, batched=False)
            ensemble_x[sample_idx,:,:,:] = model(data['x'].unsqueeze(0)).to(device='cpu')
            ensemble_y[sample_idx,:,:,:] = data['y'].squeeze().to(device='cpu')
    
        MMD += (1./float(sqrtSize))*(maxMeanDiscr.eval(ensemble_x, ensemble_y)).item()

        # Clear intermediate results and free up memory
        ensemble_x.detach_()
        ensemble_y.detach_()
        gc.collect()
        print_memory_usage()

    
    return MMD


def singleEnsemble_baseline_crps(model, dataset, data_processor, set_size):
    """
    Same as baseline_crps but with one huge ensemble.
    Needs a lot of memory for big set_size.
    """

    model.eval()

    ensemble_x = torch.empty((set_size,2,128,128))
    print(ensemble_x.shape)
    ensemble_y = torch.empty((set_size,2,128,128))

    NO_crps = 0.0
    Num_crps = 0.0

    for sample_idx in range(set_size):
        data = dataset[sample_idx]
        data = data_processor.preprocess(data, batched=False)
        out = model(data['x'].unsqueeze(0)).to(device='cpu')
        out, data = data_processor.postprocess(out, data)
        ensemble_x[sample_idx,:,:,:] = out
        ensemble_y[sample_idx,:,:,:] = data['y'].squeeze().to(device='cpu')
        if sample_idx % 100 == 0:
            print(sample_idx)
    
    for i in range(set_size):
        NO_crps += (1./float(set_size))*np.mean(sr.crps_ensemble(ensemble_x.detach(), ensemble_y[i,:,:,:].detach(), axis=0), None)
        Num_crps += (1./float(set_size))*np.mean(sr.crps_ensemble(ensemble_y.detach(), ensemble_y[i,:,:,:].detach(), axis=0), None)
        if i % 100 == 0:
            print(i)
    
    # Clear intermediate results and free up memory
    ensemble_x.detach_()
    ensemble_y.detach_()
    gc.collect()
    print_memory_usage()

    return abs(NO_crps - Num_crps)


def singleEnsemble_baseline_mmd(model, dataset, data_processor, set_size):
    """
    Same as baseline_mmd but with one huge ensemble.
    Needs a lot of memory for big set_size.
    """
    #sigma = 0.0313839316368103
    sigma = -0.027283422648906708
    print(f'Gauss kernel sigma for mmd: {sigma}')
    gauss_kernel = rbf(sigma)
    maxMeanDiscr = mmd(gauss_kernel, member_dim=0, reduce_dims=None)

    model.eval()

    ensemble_x = torch.empty((set_size,2,128,128))
    print(ensemble_x.shape)
    ensemble_y = torch.empty((set_size,2,128,128))

    MMD = 0

    for sample_idx in range(set_size):
        data = dataset[sample_idx]
        data = data_processor.preprocess(data, batched=False)
        out = model(data['x'].unsqueeze(0)).to(device='cpu')
        out, data = data_processor.postprocess(out, data)
        ensemble_x[sample_idx,:,:,:] = out
        ensemble_y[sample_idx,:,:,:] = data['y'].squeeze().to(device='cpu')
        if sample_idx % 100 == 0:
            print(sample_idx)

    MMD = (1./float(set_size))*(maxMeanDiscr.eval(ensemble_x, ensemble_y)).item()


    # Clear intermediate results and free up memory
    ensemble_x.detach_()
    ensemble_y.detach_()
    gc.collect()
    print_memory_usage()

    
    return MMD