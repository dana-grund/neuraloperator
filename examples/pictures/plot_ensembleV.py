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
Script for plotting the ground truth v fields of ten samples of the first ensemble of the ensemble dataset.
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
j = 0
size = j*100 + 100
ensembleSet = ShearLayerSeriesDataset(128, size, 2, 'test', ensemble=True)
#data_processor = data_processor.to(device)

# # Load model for forward pass
# folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/correctedEns_model/tmp/"
# name = "fno_shear_n_train=40000_epoch=5_correctedEns_cpu"
# model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
# model.eval()
# print("Model loaded")
# # model.to(device)

# Forward pass of first ensemble
# num_batches = 1000 / batch_size
# print(f"Num. batches: {num_batches}")
# num_batches = int(num_batches)
# velocitiesIn, labelVelsIn = forwardPass(model, ensemble_loaders[0], num_batches, data_processor)
# velocitiesIn = np.transpose(velocitiesIn, axes=(0,1,3,2))
# labelVelsIn = np.transpose(labelVelsIn, axes=(0,1,3,2))

# velocitiesOut, labelVelsOut = forwardPass(model, ensemble_loaders[1], num_batches, data_processor)
# velocitiesOut = np.transpose(velocitiesOut, axes=(0,1,3,2))
# labelVelsOut = np.transpose(labelVelsOut, axes=(0,1,3,2))
# print("Pass done")

# plotting
fig, axs = plt.subplots(2,5, figsize=(20, 20))
fig.set_figheight(7)

# maxGlobI = 0
# maxGlob = 0

caxList = []

for i in range(10):

    #u in
    #predIn = velocitiesIn[i*100:(i+1)*100, 0, :, :]
    #labelIn = labelVelsIn[i*100:(i+1)*100, 0, :, :]

    #sigs_in_pred = np.mean(predIn, axis=0)
    #sigs_in_lab = np.mean(labelIn, axis=0)

    field = ensembleSet[j*100 + i][9]

    col = i % 5
    row = i // 5
    caxList.append(axs[row,col].imshow(field[1], extent=[0, 127, 0, 127]))
    #divider = make_axes_locatable(axs[row,col])
    #cax = divider.append_axes("right", size="6%", pad=0.06)
    fig.colorbar(caxList[-1], ax=axs[row,col], fraction=0.046, pad=0.04)
    if row == 1:
        if col == 0:
            axs[row,col].set(xlabel='x', ylabel='y')
        else:
            axs[row,col].set(xlabel='x')
    else:
        if col == 0:
            axs[row,col].set(ylabel='y')
    axs[row,col].set_title(f'Member {i+1}')

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
fig.suptitle(f"Ensemble {j+1} - Label v", fontsize=16, x=0.43, y=0.92)  # Raise the title

#cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

ticks = np.arange(0, 128, 20)
for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

#plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
fig.tight_layout(rect=[0, 0, 0.85, 0.93])

plt.savefig(f"uEnsemble{j+1}V.png")
