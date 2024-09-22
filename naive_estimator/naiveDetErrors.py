from naiveDataset import ShearLayerNaiveDataset, get_ensembleSet, get_baselineSet, get_stabilitySet
from neuralop import LpLoss, H1Loss, print_scores

"""
Dataset that serves as a naive estimator by outputing T and T-1 values instead of T and 0 values like the regular dataset.
Includes functions for acquire different types of naive estimators.
"""

# Create the losses
reduce_dims = 0
reductions = 'mean'
l2loss = LpLoss(d=2, p=2, reduce_dims=reduce_dims, reductions=reductions)
h1loss = H1Loss(d=2, reduce_dims=reduce_dims, reductions=reductions)
l1loss = LpLoss(d=2, p=1, reduce_dims=reduce_dims, reductions=reductions)

eval_losses = {'l2': l2loss, 'h1': h1loss, 'l1': l1loss}

ensemble_db = get_ensembleSet()
stability_db = get_stabilitySet()
baseline_db = get_baselineSet(10000)

baseScores_abs = {}
baseScores_rel = {}
ensScores_abs = {}
ensScores_rel = {}
stabScores_abs = {}
stabScores_rel = {}
for loss_name in eval_losses:
    #print(f'{loss_name}, {losses[loss_name]}')
    baseScores_abs[loss_name] = 0.0
    baseScores_rel[loss_name] = 0.0
    ensScores_abs[loss_name] = 0.0
    ensScores_rel[loss_name] = 0.0
    stabScores_abs[loss_name] = 0.0
    stabScores_rel[loss_name] = 0.0

# Compute abs/rel scores for all losses, averaged over all test samples
for sample_index in range(10000):
    baseData = baseline_db[sample_index]
    baseY = baseData['y'].squeeze()
    baseX = baseData['x'].squeeze()
    if sample_index < 1000:
        ensData = ensemble_db[sample_index]
        ensY = ensData['y'].squeeze()
        ensX = ensData['x'].squeeze()
        stabData = stability_db[sample_index]
        stabY = stabData['y'].squeeze()
        stabX = stabData['x'].squeeze()
    
    for loss_name in eval_losses:
        baseScores_abs[loss_name] += (1./10000.)*(eval_losses[loss_name].abs(baseX, baseY)).item()
        baseScores_rel[loss_name] += (1./10000.)*(eval_losses[loss_name](baseX, baseY)).item()
        if sample_index < 1000:
            ensScores_abs[loss_name] += (1./1000.)*(eval_losses[loss_name].abs(ensX, ensY)).item()
            ensScores_rel[loss_name] += (1./1000.)*(eval_losses[loss_name](ensX, ensY)).item()
            stabScores_abs[loss_name] += (1./1000.)*(eval_losses[loss_name].abs(stabX, stabY)).item()
            stabScores_rel[loss_name] += (1./1000.)*(eval_losses[loss_name](stabX, stabY)).item()

print('\nOn baseline data')
print_scores(scores_abs=baseScores_abs, scores_rel=baseScores_rel, reductions=reductions)

print('\nOn ensemble data')
print_scores(scores_abs=ensScores_abs, scores_rel=ensScores_rel, reductions=reductions)

print('\nOn stability data')
print_scores(scores_abs=stabScores_abs, scores_rel=stabScores_rel, reductions=reductions)