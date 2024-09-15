import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import random as rd
from scipy.stats import norm
import os

from neuralop.models import TFNO
from neuralop.datasets import load_shear_flow, plot_shear_flow_test

"""
Script for computing the variance over an ensembles at each point for u and v.
Prints the averages over all ten ensembles and over all points for both ensemble and stability datasets.
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

velocitiesOut, labelVelsOut = forwardPass(model, ensemble_loaders[1], num_batches, data_processor)
print("Pass done")

velDir = 0

avg_sig_pred_in = 0
avg_sig_lab_in = 0
avg_sig_pred_out = 0
avg_sig_lab_out = 0

for i in range(10):
    for x_idx in range(128):
        for y_idx in range(128):
            predUIn = velocitiesIn[i*100:(i+1)*100, velDir, y_idx, x_idx]
            labelUIn = labelVelsIn[i*100:(i+1)*100, velDir, y_idx, x_idx]

            avg_sig_pred_in += np.std(predUIn)
            avg_sig_lab_in += np.std(labelUIn)

            predUOut = velocitiesOut[i*100:(i+1)*100, velDir, y_idx, x_idx]
            labelUOut = labelVelsOut[i*100:(i+1)*100, velDir, y_idx, x_idx]

            avg_sig_pred_out += np.std(predUOut)
            avg_sig_lab_out += np.std(labelUOut)
    

avg_sig_pred_in /= (10*128*128)
avg_sig_pred_out /= (10*128*128)
avg_sig_lab_in /= (10*128*128)
avg_sig_lab_out /= (10*128*128)

print("\n\nIn distribution u:\n")
print(f"Label: {avg_sig_lab_in}\n")
print(f"Predicted: {avg_sig_pred_in}")

print("\n\nOut of distribution u:\n")
print(f"Label: {avg_sig_lab_out}\n")
print(f"Predicted: {avg_sig_pred_out}")

velDir = 1

avg_sig_pred_in = 0
avg_sig_lab_in = 0
avg_sig_pred_out = 0
avg_sig_lab_out = 0

for i in range(10):
    for x_idx in range(128):
        for y_idx in range(128):
            predUIn = velocitiesIn[i*100:(i+1)*100, velDir, y_idx, x_idx]
            labelUIn = labelVelsIn[i*100:(i+1)*100, velDir, y_idx, x_idx]

            avg_sig_pred_in += np.var(predUIn)
            avg_sig_lab_in += np.var(labelUIn)

            predUOut = velocitiesOut[i*100:(i+1)*100, velDir, y_idx, x_idx]
            labelUOut = labelVelsOut[i*100:(i+1)*100, velDir, y_idx, x_idx]

            avg_sig_pred_out += np.var(predUOut)
            avg_sig_lab_out += np.var(labelUOut)
    

avg_sig_pred_in /= (10*128*128)
avg_sig_pred_out /= (10*128*128)
avg_sig_lab_in /= (10*128*128)
avg_sig_lab_out /= (10*128*128)

print("\n\nIn distribution v:\n")
print(f"Label: {avg_sig_lab_in}\n")
print(f"Predicted: {avg_sig_pred_in}")

print("\n\nOut of distribution v:\n")
print(f"Label: {avg_sig_lab_out}\n")
print(f"Predicted: {avg_sig_pred_out}")