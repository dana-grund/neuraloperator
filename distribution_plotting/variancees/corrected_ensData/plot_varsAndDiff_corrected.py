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
Script for plotting the differences (ground truth - prediction) in variance in u and v over the first ensemble of the ensemble dataset.
Here the difference is not taken absolute. Negative means the prediction is too big, positive means the prediction is too small.
The ground truth and the prediction are plotted next to the difference for comparison.
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
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/correctedEns_model/"
name = "fno_shear_n_train=40000_epoch=5_correctedEns_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
model.eval()
print("Model loaded")
# model.to(device)

# Forward pass of first ensemble
num_batches = 1000 / batch_size
print(f"Num. batches: {num_batches}")
num_batches = int(num_batches)+1
velocitiesIn, labelVelsIn = forwardPass(model, ensemble_loaders[0], num_batches, data_processor)
velocitiesIn = np.transpose(velocitiesIn, axes=(0,1,3,2))
labelVelsIn = np.transpose(labelVelsIn, axes=(0,1,3,2))

# velocitiesOut, labelVelsOut = forwardPass(model, ensemble_loaders[1], num_batches, data_processor)
# velocitiesOut = np.transpose(velocitiesOut, axes=(0,1,3,2))
# labelVelsOut = np.transpose(labelVelsOut, axes=(0,1,3,2))
print("Pass done")

ensemble = 0

# plotting
# plotting
fig, axs = plt.subplots(2,3, figsize=(15, 9))

maxGlobI = 0
maxGlob = 0

caxList = []

i = ensemble

#u in
predIn = velocitiesIn[i*100:(i+1)*100, 0, :, :]
labelIn = labelVelsIn[i*100:(i+1)*100, 0, :, :]

sigs_in_pred = np.var(predIn, axis=0)
sigs_in_lab = np.var(labelIn, axis=0)

sigs_in_err = sigs_in_lab - sigs_in_pred

# Ground truth sig
caxList.append(axs[0,0].imshow(sigs_in_lab, extent=[0, 127, 0, 127], cmap='YlGn'))
caxList.append(caxList[-1])
divider = make_axes_locatable(axs[0,0])
cax = divider.append_axes("right", size="7%", pad=0.1)
fig.colorbar(caxList[-1], cax=cax)

# Sig pred
caxList.append(axs[0,1].imshow(sigs_in_pred, extent=[0, 127, 0, 127], cmap='YlGn'))
caxList.append(caxList[-1])
divider = make_axes_locatable(axs[0,1])
cax = divider.append_axes("right", size="7%", pad=0.1)
fig.colorbar(caxList[-1], cax=cax)

# Sig diff
Max = np.max(sigs_in_err)
Min = np.min(sigs_in_err)
absMax = np.max([abs(Max), abs(Min)])
caxList.append(axs[0,2].imshow(sigs_in_err, extent=[0, 127, 0, 127], cmap='seismic', vmin=-absMax, vmax=absMax))
caxList.append(caxList[-1])
divider = make_axes_locatable(axs[0,2])
cax = divider.append_axes("right", size="7%", pad=0.1)
fig.colorbar(caxList[-1], cax=cax)

axs[0,0].set(ylabel='y')
axs[0,0].set_title('Variance - Ground Truth u')
axs[0,1].set_title('Variance - Predicted u')
axs[0,2].set_title('Variance Difference - u')

#v in
predIn = velocitiesIn[i*100:(i+1)*100, 1, :, :]
labelIn = labelVelsIn[i*100:(i+1)*100, 1, :, :]

sigs_in_pred = np.var(predIn, axis=0)
sigs_in_lab = np.var(labelIn, axis=0)

sigs_in_err = sigs_in_lab - sigs_in_pred

# Ground truth sig
caxList.append(axs[1,0].imshow(sigs_in_lab, extent=[0, 127, 0, 127], cmap='YlGn'))
caxList.append(caxList[-1])
divider = make_axes_locatable(axs[1,0])
cax = divider.append_axes("right", size="7%", pad=0.1)
fig.colorbar(caxList[-1], cax=cax)

# Sig pred
caxList.append(axs[1,1].imshow(sigs_in_pred, extent=[0, 127, 0, 127], cmap='YlGn'))
caxList.append(caxList[-1])
divider = make_axes_locatable(axs[1,1])
cax = divider.append_axes("right", size="7%", pad=0.1)
fig.colorbar(caxList[-1], cax=cax)

# Sig diff
Max = np.max(sigs_in_err)
Min = np.min(sigs_in_err)
absMax = np.max([abs(Max), abs(Min)])
caxList.append(axs[1,2].imshow(sigs_in_err, extent=[0, 127, 0, 127], cmap='seismic', vmin=-absMax, vmax=absMax))
caxList.append(caxList[-1])
divider = make_axes_locatable(axs[1,2])
cax = divider.append_axes("right", size="7%", pad=0.1)
fig.colorbar(caxList[-1], cax=cax)

axs[1,0].set(ylabel='y', xlabel='x')
axs[1,1].set(xlabel='x')
axs[1,2].set(xlabel='x')
axs[1,0].set_title('Variance - Ground Truth v')
axs[1,1].set_title('Variance - Predicted v')
axs[1,2].set_title('Variance Difference - v')

fig.subplots_adjust(hspace=0.1, wspace=0.2, top=0.9)
fig.suptitle("Ensemble 1, Variances - Ensemble Data", fontsize=16, x=0.43, y=0.91)  # Raise the title

#cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

ticks = np.arange(0, 128, 20)
for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

#plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
fig.tight_layout(rect=[0, 0, 0.85, 0.92])

plt.savefig(f"ensPlots/VarsAndDiff{ensemble+1}_corrected.png")
