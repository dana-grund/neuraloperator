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
Script for computing the relative differnce between mean over the members of ensemble 10
of the predicted velocities and the ground truth velocities.
Here the difference is not taken absolute. Negative means the prediction is too big, positive means the prediction is too small.
Averaged over all points and all ensembles.
Since here the target is the baseline dataset, the ensembles are packs of 100 samples taken in order of the dataset.
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
n_tests = 1000
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
velocitiesIn, labelVelsIn = forwardPass(model, test_loaders[128], num_batches, data_processor)
velocitiesIn = np.transpose(velocitiesIn, axes=(0,1,3,2))
labelVelsIn = np.transpose(labelVelsIn, axes=(0,1,3,2))

# velocitiesOut, labelVelsOut = forwardPass(model, ensemble_loaders[1], num_batches, data_processor)
# velocitiesOut = np.transpose(velocitiesOut, axes=(0,1,3,2))
# labelVelsOut = np.transpose(labelVelsOut, axes=(0,1,3,2))
print("Pass done")

i=9

#u in
predIn = velocitiesIn[i*100:(i+1)*100, 1, :, :]
labelIn = labelVelsIn[i*100:(i+1)*100, 1, :, :]

sigs_in_pred = np.mean(predIn, axis=0)
sigs_in_lab = np.mean(labelIn, axis=0)

sigs_in_err = (sigs_in_lab - sigs_in_pred) / np.abs(sigs_in_lab)

print("\nRelative mean errors in v, ensemble 10 (positive->too small, negative->too big):\n")

for x in range(128):
    for y in range(128):
        print(f"({x}, {y}): {sigs_in_err[x,y]}")

m = np.unravel_index(np.argmax(sigs_in_err), sigs_in_err.shape)
print(f"\nMax: ({m[0]}, {m[1]}): {sigs_in_err[m]}")
