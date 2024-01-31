"""
Training a TFNO on the shear layer experiment
=============================================

"""

# %%
# 


import torch
import matplotlib.pyplot as plt
import sys
import os

from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_shear_flow
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = 'cpu'
folder = '/cluster/work/climate/dgrund/git/dana-grund/neuraloperator/examples/plot_FNO_shear'

# %%
# Loade the Navier--Stokes dataset
train_loader, test_loaders, data_processor = load_shear_flow(
        n_train=10,             # 1000
        batch_size=32, 
        train_resolution=64,
        test_resolutions=[64],  # [64,128], 
        n_tests=[16],           # [50, 10],
        test_batch_sizes=[32],  # [32, 32],
        positional_encoding=True
)
data_processor = data_processor.to(device)


# %%
# Create a tensorized FNO model
model = TFNO(
            n_modes=(16, 16),
            in_channels=4,
            out_channels=2,
            hidden_channels=32, 
            projection_channels=64, 
            factorization='tucker', 
            rank=0.42)
# in_channels = 2 physical variables + 2 positional encoding
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                            lr=8e-3, 
                            weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Create the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {'l2': l2loss} # {'h1': h1loss, 'l2': l2loss}


# %%
# Summary
print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
trainer = Trainer(model=model, n_epochs=2, # 20
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model
trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)
model.save_checkpoint(save_folder=folder, save_name='example_fno_shear')

# %%
# Plot the prediction, and compare with the ground-truth 
# Note that this is a minimal working example for debugging only
# In practice we would train a Neural Operator on one or multiple GPUs

test_db = test_loaders[64].dataset
model = TFNO.from_checkpoint(save_folder=folder, save_name='example_fno_shear')

n_plot = 5
fig = plt.figure(figsize=(7, 2*n_plot))
for index in range(n_plot):
    
    data = test_db[index]
    data = data_processor.preprocess(data, batched=False)

    x = data['x'].unsqueeze(0)
    out = model(x)                  # input u and v

    x = data['x'][0,:,:]            # plot u component only
    y = data['y'].squeeze()[0,:,:]  # plot u component only
    
    ax = fig.add_subplot(n_plot, 3, index*3 + 1)
    ax.imshow(x, cmap='gray')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(n_plot, 3, index*3 + 2)
    ax.imshow(y)
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(n_plot, 3, index*3 + 3)
    ax.imshow(out.squeeze()[0,:,:].detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.savefig(
    os.path.join(folder,'fig-example_shear.png')
)
fig.show()
 