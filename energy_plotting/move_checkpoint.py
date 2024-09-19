import torch
import os

from neuralop.models import TFNO

"""
Move a model checkpoint from GPU to CPU
"""

if torch.cuda.is_available():
    
    device = torch.device("cuda")
    
    # Load model for forward pass
    folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/examples/plot_FNO_shear/"
    name = "fno_shear_n_train=40000_epoch=5_correctedEns"
    model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
    
    device = torch.device("cpu")
    model = model.to(device)
    
    # Save model in cpu context
    folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/correctedEns_model/"
    name = "fno_shear_n_train=40000_epoch=5_correctedEns_cpu"
    
    model.save_checkpoint(save_folder=folder, save_name=name)
    
else:
    print("GPU access is necessary for this script!")