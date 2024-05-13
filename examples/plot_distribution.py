import numpy as np
import matplotlib.pyplot as plt
import random as rd
from datetime import datetime
import torch
import math
from scipy.stats import norm
from scipy.stats import lognorm

from neuralop.models import TFNO
from neuralop.datasets import ShearLayerDataset

n_samples = 100
ensemble_size = 100
n_plots = 10

rd.seed(datetime.now().timestamp())

#print(f'Random u locatin: ({x_idx}, {y_idx})')

#train_loader, test_loaders, ensemble_loader, data_processor = load_shear_flow(
#        n_train=n_samples,             # 40_000
#        batch_size=32, 
#        train_resolution=res,
#        test_resolutions=[128],  # [64,128], 
#        n_tests=[10000],           # [10_000, 10_000],
#        test_batch_sizes=[32],  # [32, 32],
#        positional_encoding=True,
#)


ensemble_db = ShearLayerDataset(
    128,
    10000,
    2,
    which='test',
    T=20,
    ensemble=True
)

#model = TFNO.from_checkpoint(save_folder=folder, save_name='example_fno_shear')

labels_u = np.zeros(ensemble_size)

fig, axs = plt.subplots(n_plots, 2, figsize=(16,19), layout="constrained")
fig.suptitle('Regular (left) and log (right, abs(min)+1 Value added) distributions', fontsize=16)
#fig.tight_layout(pad=3.0, h_pad=3.0)
#fig.set_size_inches(7,9)

for k in range(n_plots):
    
    x_idx = rd.randint(0, 127)
    y_idx = rd.randint(0, 127)

    for i in range(ensemble_size):
        labels_u[i] = ensemble_db[k*ensemble_size + i]['y'][0,x_idx,y_idx]
        
    # z score normalisation
    mu = np.mean(labels_u)
    std = np.std(labels_u)
    normal_labels = (labels_u - mu) / std
    
    # Shift to positive for log
    minU = np.min(normal_labels)
    normal_labels += abs(minU)
    
    counts_lab, bins_lab = np.histogram(normal_labels, bins=20, density=True)
    axs[k,0].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab)
    axs[k,0].set_title(f'Random u locatin ({x_idx}, {y_idx}) distribution in label ensemble')
    axs[k,0].set(xlabel='u', ylabel='Density')
   
    mu = np.mean(normal_labels)
    std = np.std(normal_labels)
    xmin, xmax = axs[k,0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[k,0].plot(x, normal, 'k', linewidth=2)
    # lognorm
    shape, loc, scale = lognorm.fit(normal_labels)
    lgnrm = lognorm.pdf(x, shape, loc, scale)
    axs[k,0].plot(x, lgnrm, 'k', linewidth=2)
    
    minU = np.min(labels_u)
    labels_u = np.log(labels_u + abs(minU) + 1.0)
    counts_lab, bins_lab = np.histogram(labels_u, bins=20, density=True)
    axs[k,1].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab)
    axs[k,1].set_title(f'Random u locatin ({x_idx}, {y_idx}) distribution in label ensemble')
    axs[k,1].set(xlabel='u', ylabel='Density')
    
    mu = np.mean(labels_u)
    std = np.std(labels_u)
    xmin, xmax = axs[k,1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[k,1].plot(x, normal, 'k', linewidth=2)

plt.savefig("ensemble_distribution.png")