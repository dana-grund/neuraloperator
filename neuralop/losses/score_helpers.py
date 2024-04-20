import torch
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np


def compute_deterministic_scores(
    test_db,
    model,
    data_processor,
    losses
):
    """
    Compute scores based on a dictionary of losses ('losses').
    Outputs two dictionaries with the abs and rel scores of all the losses.
    """ 
    
    scores_abs = {}
    scores_rel = {}
    for loss_name in losses:
        #print(f'{loss_name}, {losses[loss_name]}')
        scores_abs[loss_name] = 0.0
        scores_rel[loss_name] = 0.0
    
    # Compute abs/rel scores for all losses, averaged over all test samples
    n = test_db.__len__()
    for sample_index in range(n):
        data = test_db[sample_index]
        data = data_processor.preprocess(data, batched=False)
        y = data['y'].squeeze()
        out = model(data['x'].unsqueeze(0))
        out = out.squeeze()
        
        for loss_name in losses:
            scores_abs[loss_name] += (1./n)*(losses[loss_name].abs(out, y)).item()
            scores_rel[loss_name] += (1./n)*(losses[loss_name](out, y)).item()
            
    return scores_abs, scores_rel

def print_scores(
    scores_abs=None,
    scores_rel=None,
    reductions=None,
    probScores=None,
):
    """
    Prints rel, abs and probabilistic scores.
    Input are dictionaries, except for'reductions', which can be a string or list/tuple of strings.
    Only prints dictionaries which are not None.
    """
    if scores_abs is not None or scores_rel is not None:
        if reductions is not None:
            print(f"\nDeterministic Scores (reductions: {reductions}):")
        else:
            print(f"\nDeterministic Scores:")
        if scores_abs is not None:
            det_row1 = list(scores_abs.keys())
        else:
            det_row1 = list(scores_rel.keys())
        det_row1.insert(0, '-')
        if scores_abs is not None:
            det_row2 = list(scores_abs.values())
            det_row2.insert(0, "Absolute")
        if scores_rel is not None:
            det_row3 = list(scores_rel.values())
            det_row3.insert(0, "Relative")
        if scores_abs is not None:
            if scores_rel is not None:
                det_table = [det_row1, det_row2, det_row3]
            else:
                det_table = [det_row1, det_row2]
        else:
            det_table = [det_row1, det_row3]
        print(tabulate(det_table, headers='firstrow', tablefmt='fancy_grid'))
    
    if probScores is not None:
        print("\nProbabilistic Scores:")
        prob_row1 = list(probScores.keys())
        prob_row2 = list(probScores.values())
        prob_table = [prob_row1, prob_row2]
        print(tabulate(prob_table, headers='firstrow', tablefmt='fancy_grid'))
    
def plot_scores(
    scores_det=None,
    scores_prob=None
):
    """
    """
    if scores_det is None and scores_prob is not None:
        fig, axs = plt.subplots(1)
        fig.tight_layout(pad=3.0)
        fig.set_size_inches(8,7)
        probMat = dictToArray(scores_prob=scores_prob)
        for scoreIdx in range(probMat.shape[0]):
            x = np.arange(probMat.shape[1])
            axs.plot(x, probMat[scoreIdx,:], label=list(scores_prob[0].keys())[scoreIdx])
        axs.set(xlabel='Epoch', ylabel='Error')
        axs.set_xticks(range(probMat.shape[1]))
        axs.legend()
        axs.set_title('Probabilistic errors')
        
    elif scores_det is not None and scores_prob is None:
        fig, axs = plt.subplots(1)
        fig.tight_layout(pad=3.0)
        fig.set_size_inches(8,7)
        detMat = dictToArray(scores_det=scores_det)
        for scoreIdx in range(detMat.shape[0]):
            x = np.arange(detMat.shape[1])
            axs.plot(x, detMat[scoreIdx,:], label=list(scores_det[0].keys())[scoreIdx])
        axs.set(xlabel='Epoch', ylabel='Error')
        axs.set_xticks(range(detMat.shape[1]))
        axs.legend()
        axs.set_title('Deterministic errors')
        
    elif scores_det is not None and scores_prob is not None:
        fig, axs = plt.subplots(2)
        fig.tight_layout(pad=3.0)
        fig.set_size_inches(8,7)
        detMat, probMat = dictToArray(scores_det=scores_det, scores_prob=scores_prob)
        for scoreIdx in range(detMat.shape[0]):
            x = np.arange(detMat.shape[1])
            axs[0].plot(x, detMat[scoreIdx,:], label=list(scores_det[0].keys())[scoreIdx])
        for scoreIdx in range(probMat.shape[0]):
            x = np.arange(probMat.shape[1])
            axs[1].plot(x, probMat[scoreIdx,:], label=list(scores_prob[0].keys())[scoreIdx])
        for ax in axs.flat:
            ax.set(xlabel='Epoch', ylabel='Error')
            ax.set_xticks(range(detMat.shape[1]))
            ax.legend()
        axs[0].set_title('Deterministic errors')
        axs[1].set_title('Probabilistic errors')
    plt.savefig("train_errors.png")
    
    
def dictToArray(
    scores_det=None,
    scores_prob=None
):
    """
    """
    if scores_det is not None and scores_prob is not None:
        detMat = np.zeros((len(scores_det[0].values()), len(scores_det)))
        probMat = np.zeros((len(scores_prob[0].values()), len(scores_prob)))
        for epoch in range(len(scores_det)):
            scoreIdxDet = 0
            scoreIdxProb = 0
            for detScore in scores_det[epoch].values():
                detMat[scoreIdxDet, epoch] = detScore
                scoreIdxDet += 1
            for probScore in scores_prob[epoch].values():
                probMat[scoreIdxProb, epoch] = probScore
                scoreIdxProb += 1
        
        return detMat, probMat
        
    elif scores_det is None and scores_prob is not None:
        probMat = np.zeros((len(scores_prob[0].values()), len(scores_prob)))
        for epoch in range(len(scores_prob)):
            scoreIdx = 0
            for probScore in scores_prob[epoch].values():
                probMat[scoreIdx, epoch] = probScore
                scoreIdx += 1
                
        return probMat
                
    elif scores_det is not None and scores_prob is None:
        detMat = np.zeros((len(scores_det[0].values()), len(scores_det)))
        for epoch in range(len(scores_det)):
            scoreIdx = 0
            for detScore in scores_det[epoch].values():
                detMat[scoreIdx, epoch] = detScore
                scoreIdx += 1
                
        return detMat
