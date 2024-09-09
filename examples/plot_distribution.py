"""
Plot horizontal velocity distributions of 10 different sample labels from ensemble dataset.
Over every distribution plot a normal distribution.
In second column plot log of velocities and a lognormal distribution.
"""
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

# RNG
rd.seed(datetime.now().timestamp())

# Initialize ensemble dataset
ensemble_db = ShearLayerDataset(
    128,
    10000,
    2,
    which='test',
    T=20,
    ensemble=True
)

labels_u = np.zeros(ensemble_size)

fig, axs = plt.subplots(n_plots, 2, figsize=(16,19), layout="constrained")
fig.suptitle('Regular (left) and log (right, abs(min)+1 Value added) distributions', fontsize=16)

for k in range(n_plots):
    
    # Generate random point
    x_idx = rd.randint(0, 127)
    y_idx = rd.randint(0, 127)

    # Get u labels at this point
    for i in range(ensemble_size):
        labels_u[i] = ensemble_db[k*ensemble_size + i]['y'][0,x_idx,y_idx]
        
    # z score standardization
    mu = np.mean(labels_u)
    std = np.std(labels_u)
    normal_labels = (labels_u - mu) / std
    
    # Shift to positive for log
    minU = np.min(normal_labels)
    normal_labels += abs(minU)
    
    # Plot histogram
    counts_lab, bins_lab = np.histogram(normal_labels, bins=20, density=True)
    axs[k,0].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab)
    axs[k,0].set_title(f'Random u locatin ({x_idx}, {y_idx}) distribution in label ensemble')
    axs[k,0].set(xlabel='u', ylabel='Density')
    
    # plot normal distribution
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
    
    # plot log historgam
    minU = np.min(labels_u)
    labels_u = np.log(labels_u + abs(minU) + 1.0)
    counts_lab, bins_lab = np.histogram(labels_u, bins=20, density=True)
    axs[k,1].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab)
    axs[k,1].set_title(f'Random u locatin ({x_idx}, {y_idx}) distribution in label ensemble')
    axs[k,1].set(xlabel='u', ylabel='Density')
    
    # plot lognormal distribution
    mu = np.mean(labels_u)
    std = np.std(labels_u)
    xmin, xmax = axs[k,1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[k,1].plot(x, normal, 'k', linewidth=2)

plt.savefig("ensemble_distribution.png")