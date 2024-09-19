import torch
from neuralop.datasets import load_shear_flow, plot_shear_flow_test
from neuralop.losses import compute_deterministic_scores, print_scores, H1Loss, LpLoss, singleEnsemble_baseline_crps
from neuralop.models import TFNO
import time

"""
Script for computing the CRPS for the baseline dataset.
Tries to compute the CRPS of one huge ensemble containing the whole dataset.
!!High memory requirement!!
"""

startSetup = time.time()

train_loader, test_loaders, ensemble_loaders, data_processor = load_shear_flow(
        n_train=10,             # 40_000
        batch_size=32, 
        train_resolution=128,
        test_resolutions=[128],  # [64,128], 
        n_tests=[10000],           # [10_000, 10_000],
        test_batch_sizes=[32],  # [32, 32],
        positional_encoding=True,
)

# Load model for forward pass
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/actualInOut/"
name = "fno_shear_n_train=40000_epoch=5_actualInOut_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
print("Model loaded")

test_db = test_loaders[128].dataset

endSetup = time.time()
startCalc = endSetup

set_size = 10000
crps = singleEnsemble_baseline_crps(model, test_db, data_processor, set_size)

endCalc = time.time()

print(f"Single Ensemble Baseline crps ({set_size}): {crps}")
print(f"Set up time: {endSetup-startSetup}, Calculation time: {endCalc-startCalc}")

f = open(f"singleEnsemble_baselineCrpsFNO{set_size}.txt", "x")
f.write(f"{crps}")
f.close()