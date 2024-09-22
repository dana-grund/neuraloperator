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

import time
import sys
import os
from naiveDataset import get_ensembleSet

"""
Script for computing errors in ensemble variances in velocities for the naive estimator on the ensemble dataset.
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

ensemble_db = get_ensembleSet()

avgUvarDiff = 0
avgVvarDiff = 0

print("\nNaive, average variance differences - ensemble set\n")

ensemble_x = np.empty((100,128,128))
ensemble_y = np.empty((100,128,128))

for i in range(10):

    for sample_idx in range(100):
        data = ensemble_db[i*100 + sample_idx]
        ensemble_x[sample_idx,:,:] = (data['x'].squeeze().to(device='cpu'))[0,:,:]
        ensemble_y[sample_idx,:,:] = (data['y'].squeeze().to(device='cpu'))[0,:,:]

    sigs_in_pred = np.var(ensemble_x, axis=0)
    sigs_in_lab = np.var(ensemble_y, axis=0)

    sigs_in_err = np.abs(sigs_in_lab - sigs_in_pred)

    ensAvgU = np.mean(sigs_in_err)

    avgUvarDiff += (1./10.)*ensAvgU

    # v
    for sample_idx in range(100):
        data = ensemble_db[i*100 + sample_idx]
        ensemble_x[sample_idx,:,:] = (data['x'].squeeze().to(device='cpu'))[1,:,:]
        ensemble_y[sample_idx,:,:] = (data['y'].squeeze().to(device='cpu'))[1,:,:]

    sigs_in_pred = np.var(ensemble_x, axis=0)
    sigs_in_lab = np.var(ensemble_y, axis=0)

    sigs_in_err = np.abs(sigs_in_lab - sigs_in_pred)

    ensAvgv = np.mean(sigs_in_err)

    avgVvarDiff += (1./10.)*ensAvgv

    # print
    print(f"Ensemble {i+1}: u: {ensAvgU}, v: {ensAvgv}")

    # col = i % 2
    # row = i // 2
    # caxList.append(axs[row,col].imshow(sigs_in_err, extent=[0, 127, 0, 127]))
    # #divider = make_axes_locatable(axs[row,col])
    # #cax = divider.append_axes("right", size="6%", pad=0.06)
    # fig.colorbar(caxList[-1], ax=axs[row,col], fraction=0.046, pad=0.04)
    # if row == 4:
    #     if col == 0:
    #         axs[row,col].set(xlabel='x', ylabel='y')
    #     else:
    #         axs[row,col].set(xlabel='x')
    # else:
    #     if col == 0:
    #         axs[row,col].set(ylabel='y')
    # axs[row,col].set_title(f'Ensemble {i+1}')

    # locMax = np.max(sigs_in_lab)
    # locMin = np.min(sigs_in_lab)
    # print(f'Max ens {i+1}: {locMax}')
    # print(f'Min ens {i+1}: {locMin}')
    # if locMax > maxGlob:
    #     maxGlob = locMax
    #     maxGlobI = i

print(f"\nOver all: u: {avgUvarDiff}, v: {avgVvarDiff}, total: {(avgUvarDiff+avgVvarDiff) / 2.}")


# # Compute global min and max
# allData = [avg_sig_errs_u_in, avg_sig_errs_u_out, avg_sig_errs_v_in, avg_sig_errs_v_out]
# minGlob = np.min([np.min(data) for data in allData])
# maxGlob = np.max([np.max(data) for data in allData])


# Adjust the space between the subplots to make room for the colorbar
#fig.subplots_adjust(hspace=0.005, wspace=0.3, right=0.85)  # Decrease hspace to reduce vertical space
# fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.9)
# fig.suptitle("Variances Error in u", fontsize=16, x=0.43, y=0.92)  # Raise the title

#cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

# ticks = np.arange(0, 128, 20)
# for ax in axs.flat:
#     ax.set_xticks(ticks)
#     ax.set_yticks(ticks)

# #plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
# fig.tight_layout(rect=[0, 0, 0.85, 0.93])

# plt.savefig("ensPlots/VarEnsDiffU_corrected.png")
