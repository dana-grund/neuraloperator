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
Script for plotting velocity distribttions over an ensemble for prediction and ground truth.
A normal distribution is laied over the histograms for comparison.
"""


def forwardPass(model, loader, num_batches, processor):
    """
    Runs requested number of batches of the dataloader through the model and returns outputs and ground truths,
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


# Load model for forward pass
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/actualInOut/"
name = "fno_shear_n_train=40000_epoch=5_actualInOut_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
model.eval()
print("Model loaded")

# Forward pass of first ensemble
num_batches = 1000 / batch_size
print(f"Num. batches: {num_batches}")
num_batches = int(num_batches)
velocities, labelVels = forwardPass(model, ensemble_loaders[0], num_batches, data_processor)
print("Pass done")

fig, axs = plt.subplots(5, figsize=(10, 15))

# coordinates of considered point
x_idx = 39
y_idx = 91

# Choose wether to plot u or v
velDir = 0 # 0->u, 1->v
if velDir == 0:
    vel = 'u'
else:
    vel = 'v'

# Plotting 5 different ensembles (the first 5)
for i in range(0, 5):

    predU = velocities[i*100:(i+1)*100, velDir, y_idx, x_idx]
    labelU = labelVels[i*100:(i+1)*100, velDir, y_idx, x_idx]

    # Prediction:
    counts_lab, bins_lab = np.histogram(predU, bins=20, density=True)
    axs[i].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab, alpha=0.5, label="Predicted", color="blue")
    
    mu = np.mean(predU)
    std = np.std(predU)
    var = np.var(predU)
    print(f'Prediction {i+1}: {var}')
    xmin, xmax = axs[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[i].plot(x, normal, linewidth=2, color="blue", label=f"Normal, std={std}")
    
    counts_lab, bins_lab = np.histogram(labelU, bins=20, density=True)
    axs[i].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab, alpha=0.5, label="Label", color="orange")
    
    # Ground truth:
    mu = np.mean(labelU)
    std = np.std(labelU)
    var = np.var(labelU)
    print(f'Label {i+1}: {var}')
    xmin, xmax = axs[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[i].plot(x, normal, linewidth=2, color="orange", label=f"Normal, std={std}")
    
    axs[i].set_title(vel + f' locatin ({x_idx}, {y_idx}) distribution, ensemble {i+1}')
    axs[i].set(xlabel=vel, ylabel='Density')
    
for ax in axs.flat:
    ax.legend()
    
plt.tight_layout()
plt.savefig(f"ensemble_distributions_" + vel + f"_In_1to5_y={y_idx}_x={x_idx}.png")



fig, axs = plt.subplots(5, figsize=(10, 15))

# Plotting 5 different ensembles (6 to 10)
for i in range(5, 10):

    predU = velocities[i*100:(i+1)*100, velDir, y_idx, x_idx]
    labelU = labelVels[i*100:(i+1)*100, velDir, y_idx, x_idx]

    # Prediction:
    counts_lab, bins_lab = np.histogram(predU, bins=20, density=True)
    axs[i-5].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab, alpha=0.5, label="Predicted", color="blue")
    
    mu = np.mean(predU)
    std = np.std(predU)
    var = np.var(predU)
    print(f'Prediction {i+1}: {var}')
    xmin, xmax = axs[i-5].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[i-5].plot(x, normal, linewidth=2, color="blue", label=f"Normal, std={std}")
    
    # Ground truth:
    counts_lab, bins_lab = np.histogram(labelU, bins=20, density=True)
    axs[i-5].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab, alpha=0.5, label="Label", color="orange")
    
    mu = np.mean(labelU)
    std = np.std(labelU)
    var = np.var(labelU)
    print(f'Label {i+1}: {var}')
    xmin, xmax = axs[i-5].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[i-5].plot(x, normal, linewidth=2, color="orange", label=f"Normal, std={std}")
    
    axs[i-5].set_title(vel + f' locatin ({x_idx}, {y_idx}) distribution, ensemble {i+1}')
    axs[i-5].set(xlabel=vel, ylabel='Density')
    
for ax in axs.flat:
    ax.legend()
    
plt.tight_layout()
plt.savefig(f"ensemble_distributions_" + vel + f"_In_6to10_y={y_idx}_x={x_idx}.png")
    