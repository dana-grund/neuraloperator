import torch
from tabulate import tabulate


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
            scores_rel[loss_name] += (1./n)*(losses[loss_name].rel(out, y)).item()
            
    return scores_abs, scores_rel

def print_scores(
    scores_abs=None,
    scores_rel=None,
    reductions=None,
    probScores=None,
):
    """
    Prints rel, abs and probabilistic scores.
    Input are dictionaries.
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
    
