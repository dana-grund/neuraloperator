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

from seriesSet import ShearLayerSeriesDataset

"""
Script for ploting v field at timestep 0 and prediction and ground truth at the final timestep and their difference.
One figure for ensemble data and one for baseline data.
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
num_batches = int(num_batches)
velocitiesIn, labelVelsIn = forwardPass(model, ensemble_loaders[0], num_batches, data_processor)
velocitiesIn = np.transpose(velocitiesIn, axes=(0,1,3,2))
labelVelsIn = np.transpose(labelVelsIn, axes=(0,1,3,2))

# Forward pass of first ensemble
num_batches = 1000 / batch_size
print(f"Num. batches: {num_batches}")
num_batches = int(num_batches)
velocitiesTest, labelVelsTest = forwardPass(model, test_loaders[128], num_batches, data_processor)
velocitiesTest = np.transpose(velocitiesTest, axes=(0,1,3,2))
labelVelsTest = np.transpose(labelVelsTest, axes=(0,1,3,2))

# velocitiesOut, labelVelsOut = forwardPass(model, ensemble_loaders[1], num_batches, data_processor)
# velocitiesOut = np.transpose(velocitiesOut, axes=(0,1,3,2))
# labelVelsOut = np.transpose(labelVelsOut, axes=(0,1,3,2))
print("Pass done")

# plotting
fig, axs = plt.subplots(1,4, figsize=(20, 10))
fig.set_figheight(7)

test_db = test_loaders[128].dataset
ensemble_db = ensemble_loaders[0].dataset


# Regular:
caxList = []

# T=1
field = np.transpose(test_db[0]['x'], axes=(0, 2, 1))
caxList.append(axs[0].imshow(field[1], extent=[0, 127, 0, 127]))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[0], fraction=0.046, pad=0.04, label='v')

# T=10
field = labelVelsTest[0, 1, :, :]
caxList.append(axs[2].imshow(field, extent=[0, 127, 0, 127]))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[1], fraction=0.046, pad=0.04, label='v')

# Prediction
field = velocitiesTest[0, 1, :, :]
caxList.append(axs[1].imshow(field, extent=[0, 127, 0, 127]))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[2], fraction=0.046, pad=0.04, label='v')

# Differnce
field = labelVelsTest[0, 1, :, :] - velocitiesTest[0, 1, :, :]
caxList.append(axs[3].imshow(field, extent=[0, 127, 0, 127]))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[3], fraction=0.046, pad=0.04, label='Difference in v')



axs[0].set(xlabel='x', ylabel='y')
axs[1].set(xlabel='x')
axs[2].set(xlabel='x')
axs[3].set(xlabel='x')
axs[0].set_title('T = 1')
axs[1].set_title('T = 10 - Label')
axs[2].set_title('T = 10 - Prediction')
axs[3].set_title('T = 10 - Difference')

# locMax = np.max(sigs_in_lab)
# locMin = np.min(sigs_in_lab)
# print(f'Max ens {i+1}: {locMax}')
# print(f'Min ens {i+1}: {locMin}')
# if locMax > maxGlob:
#     maxGlob = locMax
#     maxGlobI = i


# # Compute global min and max
# allData = [avg_sig_errs_u_in, avg_sig_errs_u_out, avg_sig_errs_v_in, avg_sig_errs_v_out]
# minGlob = np.min([np.min(data) for data in allData])
# maxGlob = np.max([np.max(data) for data in allData])


# Adjust the space between the subplots to make room for the colorbar
#fig.subplots_adjust(hspace=0.005, wspace=0.3, right=0.85)  # Decrease hspace to reduce vertical space
fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.9)
fig.suptitle("Baseline Label vs Prediction - v", fontsize=16, x=0.43, y=0.72)  # Raise the title

#cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

ticks = np.arange(0, 128, 20)
for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

#plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
fig.tight_layout(rect=[0, 0, 0.85, 0.92])

plt.savefig(f"firstLastBothDiffRegularV1.png")

# plotting
fig, axs = plt.subplots(1,4, figsize=(20, 10))
fig.set_figheight(7)

# Ensemble:
caxList = []

# T=1
field = np.transpose(ensemble_db[0]['x'], axes=(0, 2, 1))
caxList.append(axs[0].imshow(field[1], extent=[0, 127, 0, 127]))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[0], fraction=0.046, pad=0.04, label='v')

# T=10
field = labelVelsIn[0, 1, :, :]
caxList.append(axs[1].imshow(field, extent=[0, 127, 0, 127]))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[1], fraction=0.046, pad=0.04, label='v')

# Prediction
field = velocitiesIn[0, 1, :, :]
caxList.append(axs[2].imshow(field, extent=[0, 127, 0, 127]))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[2], fraction=0.046, pad=0.04, label='v')

# Prediction
field = labelVelsIn[0, 1, :, :] - velocitiesIn[0, 1, :, :]
caxList.append(axs[3].imshow(field, extent=[0, 127, 0, 127]))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[3], fraction=0.046, pad=0.04, label='Difference in v')


axs[0].set(xlabel='x', ylabel='y')
axs[1].set(xlabel='x')
axs[2].set(xlabel='x')
axs[3].set(xlabel='x')
axs[0].set_title('T = 1')
axs[1].set_title('T = 10 - Label')
axs[2].set_title('T = 10 - Prediction')
axs[3].set_title('T = 10 - Difference')

# locMax = np.max(sigs_in_lab)
# locMin = np.min(sigs_in_lab)
# print(f'Max ens {i+1}: {locMax}')
# print(f'Min ens {i+1}: {locMin}')
# if locMax > maxGlob:
#     maxGlob = locMax
#     maxGlobI = i


# # Compute global min and max
# allData = [avg_sig_errs_u_in, avg_sig_errs_u_out, avg_sig_errs_v_in, avg_sig_errs_v_out]
# minGlob = np.min([np.min(data) for data in allData])
# maxGlob = np.max([np.max(data) for data in allData])


# Adjust the space between the subplots to make room for the colorbar
#fig.subplots_adjust(hspace=0.005, wspace=0.3, right=0.85)  # Decrease hspace to reduce vertical space
fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.9)
fig.suptitle("Ensemble Label vs Prediction - v", fontsize=16, x=0.43, y=0.72)  # Raise the title

#cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

ticks = np.arange(0, 128, 20)
for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

#plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
fig.tight_layout(rect=[0, 0, 0.85, 0.92])

plt.savefig(f"firstLastBothDiffEnsembleV1.png")