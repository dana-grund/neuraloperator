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
Same as other plot_distribution.py scripts but here for the baseline dataset.
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
n_tests = 10000
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
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/correctedEns_model/"
name = "fno_shear_n_train=40000_epoch=5_correctedEns_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
model.eval()
print("Model loaded")

# Forward pass of first ensemble
num_batches = 1000 / batch_size
print(f"Num. batches: {num_batches}")
num_batches = int(num_batches) + 1
velocities, labelVels = forwardPass(model, test_loaders[128], num_batches, data_processor)
print("Pass done")
velocities = np.transpose(velocities, axes=(0,1,3,2))
labelVels = np.transpose(labelVels, axes=(0,1,3,2))

fig, axs = plt.subplots(5, figsize=(10, 15))

x_idx = 5
y_idx = 81

velDir = 1 # 0->u, 1->v
if velDir == 0:
    vel = 'u'
else:
    vel = 'v'

for i in range(0, 5):

    predU = velocities[i*100:(i+1)*100, velDir, x_idx, y_idx]
    labelU = labelVels[i*100:(i+1)*100, velDir, x_idx, y_idx]
    counts_lab, bins_lab = np.histogram(predU, bins=20, density=True)
    axs[i].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab, alpha=0.5, label="Predicted", color="blue")
    
    mu = np.mean(predU)
    std = np.std(predU)
    var = np.var(predU)
    print(f'Prediction {i+1}: {var}')
    xmin, xmax = axs[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[i].plot(x, normal, linewidth=2, color="blue", label=f"Normal, std={std:2.4f}")
    
    counts_lab, bins_lab = np.histogram(labelU, bins=20, density=True)
    axs[i].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab, alpha=0.5, label="Ground-Truth", color="orange")
    
    mu = np.mean(labelU)
    std = np.std(labelU)
    var = np.var(labelU)
    print(f'Ground-Truth {i+1}: {var}')
    xmin, xmax = axs[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[i].plot(x, normal, linewidth=2, color="orange", label=f"Normal, std={std:2.4f}")
    
    axs[i].set_title('Baseline Data - '+vel+f' Distribution, Locatin ({x_idx}, {y_idx}), Ensemble {i+1}')
    axs[i].set(xlabel=vel, ylabel='Density')
    
for ax in axs.flat:
    ax.legend()
    
plt.tight_layout()
if not os.path.exists("baselineData/" + f"{x_idx}_{y_idx}"):
    os.mkdir("baselineData/" + f"{x_idx}_{y_idx}")
plt.savefig("baselineData/" + f"{x_idx}_{y_idx}/" + "baseline_distributions_" + vel + f"_1to5_x={x_idx}_y={y_idx}.png")



fig, axs = plt.subplots(5, figsize=(10, 15))

# Second half
for i in range(5, 10):

    predU = velocities[i*100:(i+1)*100, velDir, x_idx, y_idx]
    labelU = labelVels[i*100:(i+1)*100, velDir, x_idx, y_idx]
    counts_lab, bins_lab = np.histogram(predU, bins=20, density=True)
    axs[i-5].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab, alpha=0.5, label="Predicted", color="blue")
    
    mu = np.mean(predU)
    std = np.std(predU)
    var = np.var(predU)
    print(f'Prediction {i+1}: {var}')
    xmin, xmax = axs[i-5].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[i-5].plot(x, normal, linewidth=2, color="blue", label=f"Normal, std={std:2.4f}")
    
    counts_lab, bins_lab = np.histogram(labelU, bins=20, density=True)
    axs[i-5].hist(bins_lab[:-1], bins=bins_lab, weights=counts_lab, alpha=0.5, label="Ground-Truth", color="orange")
    
    mu = np.mean(labelU)
    std = np.std(labelU)
    var = np.var(labelU)
    print(f'Ground-Truth {i+1}: {var}')
    xmin, xmax = axs[i-5].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal = norm.pdf(x, mu, std)
    axs[i-5].plot(x, normal, linewidth=2, color="orange", label=f"Normal, std={std:2.4f}")
    
    axs[i-5].set_title('Baseline Data - '+vel+f' Distribution, Locatin ({x_idx}, {y_idx}), Ensemble {i+1}')
    axs[i-5].set(xlabel=vel, ylabel='Density')
    
for ax in axs.flat:
    ax.legend()
    
plt.tight_layout()
plt.savefig("baselineData/" + f"{x_idx}_{y_idx}/" + "baseline_distributions_" + vel + f"_6to10_x={x_idx}_y={y_idx}.png")
    