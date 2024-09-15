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
Script for plotting the variance over an ensembles at each point for u and v averaged over all ten ensembles of the ensemble dataset.
plots both ground truth and prediction.
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

print("Pass done")

avg_sig_pred_u = 0
avg_sig_pred_v = 0
avg_sig_lab_u = 0
avg_sig_lab_v = 0

caxList = []

for i in range(10):

    #u
    pred = velocitiesIn[i*100:(i+1)*100, 0, :, :]
    label = labelVelsIn[i*100:(i+1)*100, 0, :, :]

    sigs_pred = np.var(pred, axis=0)
    sigs_lab = np.var(label, axis=0)

    avg_sig_lab_u += sigs_lab
    avg_sig_pred_u += sigs_pred

    #v
    pred = velocitiesIn[i*100:(i+1)*100, 1, :, :]
    label = labelVelsIn[i*100:(i+1)*100, 1, :, :]

    sigs_pred = np.var(pred, axis=0)
    sigs_lab = np.var(label, axis=0)

    avg_sig_lab_v += sigs_lab
    avg_sig_pred_v += sigs_pred


avg_sig_pred_u /= 10
avg_sig_pred_v /= 10
avg_sig_lab_u /= 10
avg_sig_lab_v /= 10

# locMax = np.max(rel_sig_errors)
#     if locMax > maxGlob:
#         maxGlob = locMax
#         maxGlobI = i


# plotting
fig, axs = plt.subplots(2,2, figsize=(10, 10))
fig.set_figwidth(9)

# Compute global min and max
maxs = np.zeros((4))
maxs[0] = np.max(avg_sig_lab_u)
maxs[1] = np.max(avg_sig_pred_u)
maxs[2] = np.max(avg_sig_lab_v)
maxs[3] = np.max(avg_sig_pred_v)
Max = np.argmax(maxs)

# u lab
caxList.append(axs[1,0].imshow(avg_sig_lab_u, extent=[0, 127, 0, 127]))
fig.colorbar(caxList[-1], ax=axs[1,0], fraction=0.046, pad=0.04)
axs[1,0].set(xlabel='x', ylabel='y')
axs[1,0].set_title(f'u Label')

# u pred
caxList.append(axs[0,0].imshow(avg_sig_pred_u, extent=[0, 127, 0, 127]))
fig.colorbar(caxList[-1], ax=axs[0,0], fraction=0.046, pad=0.04)
axs[0,0].set(xlabel='x', ylabel='y')
axs[0,0].set_title(f'u Prediction')

# v lab
caxList.append(axs[1,1].imshow(avg_sig_lab_v, extent=[0, 127, 0, 127]))
fig.colorbar(caxList[-1], ax=axs[1,1], fraction=0.046, pad=0.04)
axs[1,1].set(xlabel='x')
axs[1,1].set_title(f'v Label')

# v pred
caxList.append(axs[0,1].imshow(avg_sig_pred_v, extent=[0, 127, 0, 127]))
fig.colorbar(caxList[-1], ax=axs[0,1], fraction=0.046, pad=0.04)
axs[0,1].set(xlabel='x')
axs[0,1].set_title(f'v Prediction')


# Adjust the space between the subplots to make room for the colorbar
fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.9)  # Decrease hspace to reduce vertical space
fig.suptitle("Mean Variance", fontsize=16, x=0.5, y=0.95)  # Raise the title

#cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

#plt.colorbar(caxList[Max], cax=cbar_ax)
#fig.tight_layout(rect=[0, 0, 0.85, 0.93])
fig.tight_layout()

plt.savefig("meanVar.png")