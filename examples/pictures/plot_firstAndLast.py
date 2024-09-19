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
Script for plotting the ground truth u and v fields a samples of the baseline dataset.
Plots timesteps 0 and 9.
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
serieSetReg = ShearLayerSeriesDataset(128, size, 2, 'test')

serieSetEns = ShearLayerSeriesDataset(128, size, 2, 'test', ensemble=True)
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
fig, axs = plt.subplots(2,2, figsize=(10, 10))
fig.set_figheight(7)

# maxGlobI = 0
# maxGlob = 0

Fields = serieSetReg[size-1]

caxList = []


#u in
#predIn = velocitiesIn[i*100:(i+1)*100, 0, :, :]
#labelIn = labelVelsIn[i*100:(i+1)*100, 0, :, :]

#sigs_in_pred = np.mean(predIn, axis=0)
#sigs_in_lab = np.mean(labelIn, axis=0)

# u

field = Fields[0]
print(type(field[0]), field[0].shape)
Max = np.max(field[0].numpy())
Min = np.min(field[0].numpy())
absMax = np.max([abs(Max), abs(Min)])
caxList.append(axs[0,0].imshow(field[0].T, extent=[0, 127, 0, 127], cmap='seismic', vmin=-absMax, vmax=absMax))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[0,0], fraction=0.046, pad=0.04)

field = Fields[10]
Max = np.max(field[0].numpy())
Min = np.min(field[0].numpy())
absMax = np.max([abs(Max), abs(Min)])
caxList.append(axs[0,1].imshow(field[0].T, extent=[0, 127, 0, 127], cmap='seismic', vmin=-absMax, vmax=absMax))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[0,1], fraction=0.046, pad=0.04)
axs[0,0].set(ylabel='y')
axs[0,0].set_title('u at T = 0')
axs[0,1].set_title('u at T = 10')

# v

field = Fields[0]
Max = np.max(field[1].numpy())
Min = np.min(field[1].numpy())
absMax = np.max([abs(Max), abs(Min)])
caxList.append(axs[1,0].imshow(field[1].T, extent=[0, 127, 0, 127], cmap='seismic', vmin=-absMax, vmax=absMax))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[1,0], fraction=0.046, pad=0.04)

field = Fields[10]
Max = np.max(field[1].numpy())
Min = np.min(field[1].numpy())
absMax = np.max([abs(Max), abs(Min)])
caxList.append(axs[1,1].imshow(field[1].T, extent=[0, 127, 0, 127], cmap='seismic', vmin=-absMax, vmax=absMax))
#divider = make_axes_locatable(axs[row,col])
#cax = divider.append_axes("right", size="6%", pad=0.06)
fig.colorbar(caxList[-1], ax=axs[1,1], fraction=0.046, pad=0.04)
axs[1,0].set(xlabel='x', ylabel='y')
axs[1,1].set(xlabel='x')
axs[1,0].set_title('v at T = 0')
axs[1,1].set_title('v at T = 10')

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
fig.suptitle("Ground Truth Velocities", fontsize=16, x=0.44, y=0.91)  # Raise the title

#cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

ticks = np.arange(0, 128, 20)
for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

#plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
fig.tight_layout(rect=[0, 0, 0.85, 0.92])

plt.savefig(f"firstAndLastUV{size}.png")

# plotting
# fig, axs = plt.subplots(1,2, figsize=(10, 10))
# fig.set_figheight(7)

# # maxGlobI = 0
# # maxGlob = 0

# Fields = serieSetEns[size-1]

# caxList = []


# #u in
# #predIn = velocitiesIn[i*100:(i+1)*100, 0, :, :]
# #labelIn = labelVelsIn[i*100:(i+1)*100, 0, :, :]

# #sigs_in_pred = np.mean(predIn, axis=0)
# #sigs_in_lab = np.mean(labelIn, axis=0)

# field = Fields[0]

# caxList.append(axs[0].imshow(field[0].T, extent=[0, 127, 0, 127]))
# #divider = make_axes_locatable(axs[row,col])
# #cax = divider.append_axes("right", size="6%", pad=0.06)
# fig.colorbar(caxList[-1], ax=axs[0], fraction=0.046, pad=0.04)

# field = Fields[10]
# caxList.append(axs[1].imshow(field[0].T, extent=[0, 127, 0, 127]))
# #divider = make_axes_locatable(axs[row,col])
# #cax = divider.append_axes("right", size="6%", pad=0.06)
# fig.colorbar(caxList[-1], ax=axs[1], fraction=0.046, pad=0.04)
# axs[0].set(xlabel='x', ylabel='y')
# axs[1].set(xlabel='x')
# axs[0].set_title('T = 0')
# axs[1].set_title('T = 10')

# # locMax = np.max(sigs_in_lab)
# # locMin = np.min(sigs_in_lab)
# # print(f'Max ens {i+1}: {locMax}')
# # print(f'Min ens {i+1}: {locMin}')
# # if locMax > maxGlob:
# #     maxGlob = locMax
# #     maxGlobI = i


# # # Compute global min and max
# # allData = [avg_sig_errs_u_in, avg_sig_errs_u_out, avg_sig_errs_v_in, avg_sig_errs_v_out]
# # minGlob = np.min([np.min(data) for data in allData])
# # maxGlob = np.max([np.max(data) for data in allData])


# # Adjust the space between the subplots to make room for the colorbar
# #fig.subplots_adjust(hspace=0.005, wspace=0.3, right=0.85)  # Decrease hspace to reduce vertical space
# fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.9)
# fig.suptitle("Label u", fontsize=16, x=0.43, y=0.75)  # Raise the title

# #cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

# ticks = np.arange(0, 128, 20)
# for ax in axs.flat:
#     ax.set_xticks(ticks)
#     ax.set_yticks(ticks)

# #plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
# fig.tight_layout(rect=[0, 0, 0.85, 0.92])

# plt.savefig(f"firstAndLast_ensemble{size}.png")
