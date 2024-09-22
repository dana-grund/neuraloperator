from naiveDataset import get_ensembleSet, get_processor
from neuralop import ensemble_crps, mmd, rbf, compute_probabilistic_scores, print_scores
from neuralop.models import TFNO

"""
Script for computing probabilistic scores (CRPS and MMD) for the for the naive esimator on the enesmble dataset.
"""

ensemble_db = get_ensembleSet()
data_processor = get_processor(ensemble_db)

ensembleCrps = ensemble_crps(member_dim=0, reduce_dims=None)
    
#sigma = 0.0313839316368103
sigma = -0.027283422648906708
print(f'Gauss kernel sigma for mmd: {sigma}')
gauss_kernel = rbf(sigma)
maxMeanDiscr = mmd(gauss_kernel, member_dim=0, reduce_dims=None)

probab_scores = {'crps': ensembleCrps, 'mmd': maxMeanDiscr}

# Load model for forward pass
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/actualInOut/"
name = "fno_shear_n_train=40000_epoch=5_actualInOut_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)

probScoresIn = compute_probabilistic_scores(
    ensemble_db,
    model,
    data_processor,
    probab_scores
)

reductions = 'mean'
print_scores(reductions=reductions, probScoresIn=probScoresIn)