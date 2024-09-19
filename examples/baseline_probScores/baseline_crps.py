import torch
from neuralop.datasets import load_shear_flow, plot_shear_flow_test
from neuralop.losses import compute_deterministic_scores, print_scores, H1Loss, LpLoss, baseline_crps
from neuralop.models import TFNO
import time

"""
Script for computing the CRPS for the baseline dataset.
Computes the CRPS as the average over the 100 CRPS values of the ensembles.
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
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/correctedEns_model/"
name = "fno_shear_n_train=40000_epoch=5_correctedEns_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
print("Model loaded")

test_db = test_loaders[128].dataset

endSetup = time.time()
startCalc = endSetup

set_size = 10000
crps = baseline_crps(model, test_db, data_processor, set_size)

endCalc = time.time()

print(f"Baseline crps ({set_size}): {crps}")
print(f"Set up time: {endSetup-startSetup}, Calculation time: {endCalc-startCalc}")

f = open(f"baselineCrpsFNO{set_size}_correctedEns_new.txt", "x")
f.write(f"{crps}")
f.close()