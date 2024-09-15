import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import random as rd
from scipy.stats import norm
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

from neuralop.models import TFNO
from neuralop.datasets import load_shear_flow, plot_shear_flow_test

"""
Script for plotting the means of the relative differences/errors in u and v for all velocities of an ensemble.
Here the differences are not absolute but can be negative. Negative means prediction is too big, positive means prediction is to small.
Averages the errors of all ensembles at each point.
Plots for both ensemble and stability data.
"""

def forwardPass(model, loader, num_batches, processor):
    """
    """
    outStack = None
    labelStack = None
    i = 0
    for batch in loader:
        if i < num_batches:
            outputs = model(**processor.preprocess(batch))
            outputs, inputs = processor.postprocess(outputs, batch)
            if i == 0:
                outStack = outputs.detach().numpy()
                labelStack = batch['y'].detach().numpy()
            else:
                outStack = np.concatenate((outStack, outputs.detach().numpy()), axis=0)
                labelStack = np.concatenate((labelStack, batch['y'].detach().numpy()), axis=0)
            i += 1
        else:
            break
    return outStack, labelStack


# Load datasets
batch_size = 32
n_train = 40000
n_epochs = 5
predicted_t = 10
n_tests = 300
res=128
train_loader, test_loaders, ensemble_loaders, data_processor = load_shear_flow(
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
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/actualInOut/"
name = "fno_shear_n_train=40000_epoch=5_actualInOut_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
model.eval()
print("Model loaded")
# model.to(device)

# Forward pass of first ensemble
num_batches = 1000 / batch_size
print(f"Num. batches: {num_batches}")
num_batches = int(num_batches)
velocitiesIn, labelVelsIn = forwardPass(model, ensemble_loaders[0], num_batches, data_processor)
velocitiesIn = np.transpose(velocitiesIn, axes=(0,1,3,2))
labelVelsIn = np.transpose(labelVelsIn, axes=(0,1,3,2))

velocitiesOut, labelVelsOut = forwardPass(model, ensemble_loaders[1], num_batches, data_processor)
velocitiesOut = np.transpose(velocitiesOut, axes=(0,1,3,2))
labelVelsOut = np.transpose(labelVelsOut, axes=(0,1,3,2))
print("Pass done")

avg_sig_errs_u_in = 0
avg_sig_errs_v_in = 0
avg_sig_errs_u_out = 0
avg_sig_errs_v_out = 0

for i in range(10):

    #u in
    predIn = velocitiesIn[i*100:(i+1)*100, 0, :, :]
    labelIn = labelVelsIn[i*100:(i+1)*100, 0, :, :]

    sigs_in_pred = np.mean(predIn, axis=0)
    sigs_in_lab = np.mean(labelIn, axis=0)

    rel_sig_errors = (sigs_in_lab - sigs_in_pred) / sigs_in_lab

    avg_sig_errs_u_in += rel_sig_errors

    #v in
    predIn = velocitiesIn[i*100:(i+1)*100, 1, :, :]
    labelIn = labelVelsIn[i*100:(i+1)*100, 1, :, :]

    sigs_in_pred = np.mean(predIn, axis=0)
    sigs_in_lab = np.mean(labelIn, axis=0)

    rel_sig_errors = (sigs_in_lab - sigs_in_pred) / sigs_in_lab

    avg_sig_errs_v_in += rel_sig_errors

    #u out
    predIn = velocitiesOut[i*100:(i+1)*100, 0, :, :]
    labelIn = labelVelsOut[i*100:(i+1)*100, 0, :, :]

    sigs_in_pred = np.mean(predIn, axis=0)
    sigs_in_lab = np.mean(labelIn, axis=0)

    rel_sig_errors = (sigs_in_lab - sigs_in_pred) / sigs_in_lab

    avg_sig_errs_u_out += rel_sig_errors

    #v out
    predIn = velocitiesOut[i*100:(i+1)*100, 1, :, :]
    labelIn = labelVelsOut[i*100:(i+1)*100, 1, :, :]

    sigs_in_pred = np.mean(predIn, axis=0)
    sigs_in_lab = np.mean(labelIn, axis=0)

    rel_sig_errors = (sigs_in_lab - sigs_in_pred) / sigs_in_lab

    avg_sig_errs_v_out += rel_sig_errors

avg_sig_errs_u_in /= 10
avg_sig_errs_v_in /= 10
avg_sig_errs_u_out /= 10
avg_sig_errs_v_out /= 10


# plotting
fig, axs = plt.subplots(2,2, figsize=(10, 10))
fig.set_figwidth(9)

ticks = np.arange(0, 128, 20)

# Compute global min and max
allData = [avg_sig_errs_u_in, avg_sig_errs_u_out, avg_sig_errs_v_in, avg_sig_errs_v_out]
minGlob = np.min([np.min(data) for data in allData])
maxGlob = np.max([np.max(data) for data in allData])
print(f'Max: {maxGlob}, Min: {minGlob}')

# u in
caxuin = axs[0,0].imshow(avg_sig_errs_u_in, extent=[0, 127, 0, 127])
fig.colorbar(caxuin, ax=axs[0,0], fraction=0.046, pad=0.04)
axs[0,0].set(xlabel='x', ylabel='y')
axs[0,0].set_title(f'u, Ensemble Data')

# v in
caxvin = axs[0,1].imshow(avg_sig_errs_v_in, extent=[0, 127, 0, 127])
fig.colorbar(caxvin, ax=axs[0,1], fraction=0.046, pad=0.04)
axs[0,1].set(xlabel='x')
axs[0,1].set_title(f'v, Ensemble Data')

# u out
caxuout = axs[1,0].imshow(avg_sig_errs_u_out, extent=[0, 127, 0, 127])
fig.colorbar(caxuout, ax=axs[1,0], fraction=0.046, pad=0.04)
axs[1,0].set(xlabel='x', ylabel='y')
axs[1,0].set_title(f'u, Stability Data')

# v out
caxvout = axs[1,1].imshow(avg_sig_errs_v_out, extent=[0, 127, 0, 127])
fig.colorbar(caxvout, ax=axs[1,1], fraction=0.046, pad=0.04)
axs[1,1].set(xlabel='x')
axs[1,1].set_title(f'v, Stability Data')

for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


# Adjust the space between the subplots to make room for the colorbar
#fig.subplots_adjust(hspace=0.02, wspace=0.3, right=0.85)  # Decrease hspace to reduce vertical space
fig.subplots_adjust(hspace=0.001, wspace=0.3)
fig.suptitle("Mean Relative Mean Errors", fontsize=16, x=0.43, y=0.85)  # Raise the title

#cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

#plt.colorbar(caxvin, cax=cbar_ax)
fig.tight_layout(rect=[0, 0, 0.85, 0.93])

plt.savefig("meanRelativeMeanDiffsNotAbs.png")