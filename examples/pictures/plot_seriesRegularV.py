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
Script for plotting series of v fields of all timesteps for a sample of the regular dataset.
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
size = 1
serieSet = ShearLayerSeriesDataset(128, size, 2, 'test')
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
fig, axs = plt.subplots(4,3, figsize=(10.25, 12))

# maxGlobI = 0
# maxGlob = 0

Fields = serieSet[0]

caxList = []

for i in range(11):

    #u in
    #predIn = velocitiesIn[i*100:(i+1)*100, 0, :, :]
    #labelIn = labelVelsIn[i*100:(i+1)*100, 0, :, :]

    #sigs_in_pred = np.mean(predIn, axis=0)
    #sigs_in_lab = np.mean(labelIn, axis=0)

    field = Fields[i]
    col = i % 3
    row = i // 3
    Max = np.max(field[1].numpy())
    Min = np.min(field[1].numpy())
    absMax = np.max([abs(Max), abs(Min)])
    im = axs[row,col].imshow(np.transpose(field[1], axes=(1,0)), extent=[0, 127, 0, 127], cmap='seismic', vmin=-absMax, vmax=absMax)
    caxList.append(im)
    #divider = make_axes_locatable(axs[row,col])
    #cax = divider.append_axes("right", size="6%", pad=0.06)
    #fig.colorbar(caxList[-1], ax=axs[row,col], fraction=0.03, pad=0.04)

    divider = make_axes_locatable(axs[row, col])
    cax = divider.append_axes("right", size="7%", pad=0.1)
    fig.colorbar(im, cax=cax)

    axs[row,col].set(xlabel='x', ylabel='y')
    axs[row,col].set_title(f'T = {i}')

    # locMax = np.max(sigs_in_lab)
    # locMin = np.min(sigs_in_lab)
    # print(f'Max ens {i+1}: {locMax}')
    # print(f'Min ens {i+1}: {locMin}')
    # if locMax > maxGlob:
    #     maxGlob = locMax
    #     maxGlobI = i

# Remove the plot in the lower left corner (first subplot in the second row)
axs[3, 2].axis('off')

# # Compute global min and max
# allData = [avg_sig_errs_u_in, avg_sig_errs_u_out, avg_sig_errs_v_in, avg_sig_errs_v_out]
# minGlob = np.min([np.min(data) for data in allData])
# maxGlob = np.max([np.max(data) for data in allData])


# Adjust the space between the subplots to make room for the colorbar
#fig.subplots_adjust(hspace=0.005, wspace=0.3, right=0.85)  # Decrease hspace to reduce vertical space
fig.subplots_adjust(hspace=0.3, wspace=0.01, top=0.93)
fig.suptitle("Evolution of Ground Truth v", fontsize=16, x=0.5, y=0.98)  # Raise the title

#cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

ticks = np.arange(0, 128, 20)
for i in range(3):
    for j in range(4):
        if not (i == 0 and j == 1):
            axs[j,i].set_xticks(ticks)
            axs[j,i].set_yticks(ticks)

for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

#plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("vSeriesRegularSeismic_new.png")
