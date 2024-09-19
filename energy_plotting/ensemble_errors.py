import numpy as np
import torch
import os

from neuralop.models import TFNO
from neuralop.datasets import load_shear_flow, plot_shear_flow_test

from neuralop import LpLoss, H1Loss, MedianAbsoluteLoss, gaussian_crps, compute_probabilistic_scores, compute_deterministic_scores, print_scores

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name())
else:
    device = torch.device("cpu")

batch_size = 32
n_train = 10
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
data_processor = data_processor.to(device)

# Load model for forward pass
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/correctedEns_model/"
name = "fno_shear_n_train=40000_epoch=5_correctedEns_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
model.eval()
print("Model loaded")
model = model.to(device)

# Create the losses
reduce_dims = 0
reductions = 'mean'
l2loss = LpLoss(d=2, p=2, reduce_dims=reduce_dims, reductions=reductions)
h1loss = H1Loss(d=2, reduce_dims=reduce_dims, reductions=reductions)
l1loss = LpLoss(d=2, p=1, reduce_dims=reduce_dims, reductions=reductions)

eval_losses = {'l2': l2loss, 'h1': h1loss, 'l1': l1loss}


test_db = test_loaders[128].dataset
ensemble_db_in = ensemble_loaders[0].dataset
ensemble_db_out = ensemble_loaders[1].dataset

absScores, relScores = compute_deterministic_scores(
    test_db,
    model,
    data_processor,
    eval_losses
)

absEnsScoresIn, relEnsScoresIn = compute_deterministic_scores(
    ensemble_db_in,
    model,
    data_processor,
    eval_losses
)

absEnsScoresOut, relEnsScoresOut = compute_deterministic_scores(
    ensemble_db_out,
    model,
    data_processor,
    eval_losses
)

print('\nOn deterministic data')
print_scores(scores_abs=absScores, scores_rel=relScores, reductions=reductions)

print('\nOn ensemble data')
print_scores(scores_abs=absEnsScoresIn, scores_rel=relEnsScoresIn, reductions=reductions)

print('\nOn stability data')
print_scores(scores_abs=absEnsScoresOut, scores_rel=relEnsScoresOut, reductions=reductions)