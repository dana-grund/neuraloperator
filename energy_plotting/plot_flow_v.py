import torch
import os

from neuralop.models import TFNO
from neuralop.datasets import load_shear_flow, plot_shear_flow_test_v

"""
Script for plotting 5 predicted samples of the vertical velocity profile.
Samples from the ensemble dataset.
"""

# Load datasets
batch_size = 25
n_train = 40000
n_epochs = 5
predicted_t = 10
n_tests = 300
res=128
train_loader, test_loaders, ensemble_loader, data_processor = load_shear_flow(
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
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/"
name = "fno_shear_n_train=40000_epoch=5_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
model.eval()
print("Model loaded")
# model.to(device)

# Plot v
ensemble_db = ensemble_loader.dataset
folder = '/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/'
plot_shear_flow_test_v(
    ensemble_db,
    model,
    data_processor,
    n_plot=5,
    save_file=os.path.join(
        folder,f'fig-example_shear_n_train={n_train}_n_epochs={n_epochs}_ensemble_v.png'
    ),
)