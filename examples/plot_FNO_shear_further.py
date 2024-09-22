"""
Training a TFNO on the shear layer experiment after it has already been trained once.
=============================================

"""

# %%
# 


import torch
import argparse
import sys
import os
import time
import zarr
import numpy as np

from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_shear_flow, plot_shear_flow_test
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss, MedianAbsoluteLoss, gaussian_crps, compute_probabilistic_scores, compute_deterministic_scores, print_scores, hacky_crps, plot_scores, lognormal_crps, ensemble_crps, mmd, rbf
# Load model for forward pass folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/actualInOut/" name = "fno_shear_n_train=40000_epoch=5_actualInOut_cpu" model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
parser = argparse.ArgumentParser(description='Train FNO for 2D shear')
parser.add_argument('--ensemble', action=argparse.BooleanOptionalAction, default=False, required=False,
                    help='Use ensebmble data. Default: False')
parser.add_argument('--res', type=int, default=128, required=False,
                    help='training resolution (64 or 128). Default: 128')
parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=False, required=False,
                    help='Device to run on. Default (False): cpu')
parser.add_argument('-f', '--folder', type=str, default='plot_FNO_shear', required=False,
                    help='Where to store results.')
args = parser.parse_args()

if args.ensemble and args.res == 64:
    raise Exception('Ensemble data only available in 128.')

#device = 'cuda' if args.gpu else 'cpu'
if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda")
elif args.gpu and not torch.cuda.is_available():
    assert False, 'No GPU available'
else:
    device = torch.device("cpu")

folder = args.folder
res = args.res
ensemble = args.ensemble

# Print gpu model
if args.gpu:
    print(torch.cuda.get_device_name())
    

#zarr.consolidate_metadata("/cluster/work/climate/webesimo/data_N128.zarr")
#zarr.consolidate_metadata("/cluster/work/climate/webesimo/data_N64.zarr")

# %%
# Load the Navier--Stokes dataset
batch_size = 32
n_train = 40000
n_epochs = 5
predicted_t = 10
n_tests = 10000
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
data_processor = data_processor.to(device)


# %%
# Load FNO model
lastEpoch = 5
model = TFNO.from_checkpoint(save_folder=folder, save_name=f'fno_shear_n_train={len(train_loader.dataset)}_epoch={lastEpoch}_correctedEns')
model.to(device)
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
reduce_dims = 0
reductions = 'mean'
l2loss = LpLoss(d=2, p=2, reduce_dims=reduce_dims, reductions=reductions)
h1loss = H1Loss(d=2, reduce_dims=reduce_dims, reductions=reductions)
l1loss = LpLoss(d=2, p=1, reduce_dims=reduce_dims, reductions=reductions)
#medianloss = MedianAbsoluteLoss(d=2, reduce_dims=reduce_dims, reductions=reductions)

train_loss = h1loss
eval_losses = {'l2': l2loss, 'h1': h1loss, 'l1': l1loss} # {'h1': h1loss, 'l2': l2loss}

# Probabilistic score metrics
probab_scores = None
if ensemble:
    #crps = lognormal_crps(member_dim=0, reduce_dims=None)
    #hackCrps = hacky_crps(member_dim=0, reduce_dims=None)
    ensembleCrps = ensemble_crps(member_dim=0, reduce_dims=None)
    
    median_sigma = torch.median(ensemble_loaders[0].dataset[0]['y'][:,:,:])
    print(f'Gauss kernel sigma for mmd: {median_sigma}')
    gauss_kernel = rbf(median_sigma)
    maxMeanDiscr = mmd(gauss_kernel, member_dim=0, reduce_dims=None)
    #crossEntropy = cross_entropy(member_dim=0)

    probab_scores = {'crps': ensembleCrps, 'mmd': maxMeanDiscr}


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
trainer = Trainer(model=model, n_epochs=n_epochs, # 20
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=1,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model
initial_evaluation=True
start = time.time()
trainErrors_det, trainErrors_probIn, train_losses = trainer.train_further(train_loader=train_loader,
                                                  test_loaders=test_loaders,
                                                  optimizer=optimizer,
                                                  scheduler=scheduler, 
                                                  regularizer=False,
                                                  save_folder=folder,
                                                  ensemble_loaders=ensemble_loaders,
                                                  training_loss=train_loss,
                                                  eval_losses=eval_losses,
                                                  prob_losses=probab_scores,
                                                  loss_reductions=reductions,
                                                  initial_eval=initial_evaluation, lastEpoch=lastEpoch)
#model.save_checkpoint(save_folder=folder, save_name=f'fno_shear_n_train={len(train_loader.dataset)}_final')
end = time.time()
print(f'Training took {end-start} s.')

plot_scores(scores_det=trainErrors_det, scores_probIn=trainErrors_probIn, train_losses=train_losses, batchSize=batch_size, trainSetSize=n_train, save_folder="/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/examples/correctedEns/", initial_eval=initial_evaluation)

# # Test 
# start = time.time()
# test_db = test_loaders[128].dataset
# ensemble_dbIn = ensemble_loaders[0].dataset
# ensemble_dbOut = ensemble_loaders[1].dataset
# model = TFNO.from_checkpoint(save_folder=folder, save_name=f'fno_shear_n_train={len(train_loader.dataset)}_epoch={lastEpoch+n_epochs}_correctedEns')
# model.to(device)
# absScores, relScores = compute_deterministic_scores(
#     test_db,
#     model,
#     data_processor,
#     eval_losses
# )

# if ensemble:
#     probScoresIn = compute_probabilistic_scores(
#         ensemble_dbIn,
#         model,
#         data_processor,
#         probab_scores
#     )
#     probScoresOut = compute_probabilistic_scores(
#         ensemble_dbOut,
#         model,
#         data_processor,
#         probab_scores
#     )

#     print_scores(scores_abs=absScores, scores_rel=relScores, reductions=reductions, probScoresIn=probScoresIn, probScoresOut=probScoresOut)

# else:
#     print_scores(scores_abs=absScores, scores_rel=relScores, reductions=reductions)
# end = time.time()
# print(f'Evaluation took {end-start} s.')

# # %%
# # Plot the prediction, and compare with the ground-truth 
# # Note that this is a minimal working example for debugging only
# # In practice we would train a Neural Operator on one or multiple GPUs
# plot_shear_flow_test(
#     test_db,
#     model,
#     data_processor,
#     n_plot=5,
#     save_file=os.path.join(
#         folder,f'fig-example_shear_n_train={n_train}_n_epochs={n_epochs}_correctedEns.png'
#     ),
# )

# if ensemble:
#     # Once again for ensemble
#     plot_shear_flow_test(
#         ensemble_dbIn,
#         model,
#         data_processor,
#         n_plot=5,
#         save_file=os.path.join(
#             folder,f'fig-example_shear_n_train={n_train}_n_epochs={n_epochs}_correctedEns_ensembleIn.png'
#         ),
#     )

#     # Once again for ensemble
#     plot_shear_flow_test(
#         ensemble_dbOut,
#         model,
#         data_processor,
#         n_plot=5,
#         save_file=os.path.join(
#             folder,f'fig-example_shear_n_train={n_train}_n_epochs={n_epochs}_correctedEns_ensembleOut.png'
#         ),
#     )

