import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math

from neuralop.models import TFNO
from neuralop.datasets import load_shear_flow, plot_shear_flow_test

from neuralop import LpLoss, H1Loss, MedianAbsoluteLoss, gaussian_crps, compute_probabilistic_scores, compute_deterministic_scores, print_scores

from plot_spectra import forwardPass, transformTotEnergie, transformOneCompEnergy, plot2DSpectrum, plot1DSpectrum, plotMulti2DSpectrum, plotMulti1DSpectrum

"""
Script for plotting 2D spectra of total kinetic energy for two ensemble means of the baseline dataset and a member from each of the ensembles.
"""
        
# Load datasets
batch_size = 25
n_train = 40000
n_epochs = 5
predicted_t = 10
n_tests = 300
res=128
train_loader, test_loaders, ensemble_loader, data_processor = load_shear_flow(
        n_train=n_train,             # 40_000
        batch_size=batch_size, 
        train_resolution=res,
        test_resolutions=[128],  # [64,128], 
        n_tests=[n_tests],           # [10_000, 10_000],
        test_batch_sizes=[batch_size],  # [32, 32],
        positional_encoding=True,
        T=predicted_t
)
#data_processor = data_processor.to(device)

# Load model for forward pass
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/correctedEns_model/"
name = "fno_shear_n_train=40000_epoch=5_correctedEns_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
model.eval()
print("Model loaded")
# model.to(device)

# Forward pass of first ensemble
num_batches = 100 / batch_size
print(f"Num. batches: {num_batches}")
num_batches = int(num_batches) + 1
velocities, labelVels = forwardPass(model, test_loaders[res], num_batches, data_processor)
print("Pass done")
print(velocities.shape)

sqVelocities = np.square(velocities)
sqLabelVels = np.square(labelVels)

# Get fourier transforms for total energy
transforms = transformTotEnergie(sqVelocities)
labelTransf = transformTotEnergie(sqLabelVels)

# Get sample frequencies
frequencies = np.fft.fftfreq(transforms[0].shape[1])
frequencies = np.fft.fftshift(frequencies)

diffMeans, diffFirsts = plotMulti2DSpectrum(transforms, frequencies, 'Total', labelTransf=labelTransf, regularData=True)
