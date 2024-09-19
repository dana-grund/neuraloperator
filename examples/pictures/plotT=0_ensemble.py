import numpy as np
import matplotlib.pyplot as plt
import os
import torch


from neuralop.datasets import load_shear_flow

"""
Script for plotting u and v fields of a sample from the stability dataset at timestep 0.
"""

batch_size = 32
n_train = 10
n_epochs = 5
predicted_t = 10
n_tests = 10
res = 128
train_loader, test_loaders, ensemble_loaders, data_processor = load_shear_flow(
        n_train=n_train,             # 40_000
        batch_size=batch_size, 
        train_resolution=res,
        test_resolutions=[128],  # [64,128], 
        n_tests=[n_tests],           # [10_000, 10_000],
        test_batch_sizes=[32],  # [32, 32],
        positional_encoding=True,
        T=predicted_t
)

fig, axs = plt.subplots(nrows=1, ncols=2)
fig.set_figwidth(9)

data = ensemble_loaders[1].dataset[0]
data = data_processor.preprocess(data, batched=False)
u = torch.transpose(data['x'][0,:,:], 0, 1)
v = torch.transpose(data['x'][1,:,:], 0, 1)

cax = axs[0].imshow(u, origin="lower")
axs[0].set_title('u')
axs[0].set(xlabel='x', ylabel='y')
axs[1].imshow(v, origin="lower")
axs[1].set_title('v')
axs[1].set(xlabel='x')

# Adjust the space between the subplots to make room for the colorbar
fig.subplots_adjust(right=0.85)

cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

fig.suptitle("Initial velocities", fontsize=16, x=0.44, y=0.87)
plt.colorbar(cax, cax=cbar_ax)
fig.tight_layout(rect=[0, 0, 0.85, 0.93])

plt.savefig("initialsEnsemble.png")

#plt.colorbar()

