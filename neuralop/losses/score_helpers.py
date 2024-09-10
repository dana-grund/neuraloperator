import torch
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import math
import os


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
    probScoresIn=None,
    probScoresOut=None,
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
            print("\nDeterministic Scores:")
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
    
    if probScoresIn is not None or probScoresOut is not None:
        print("\nProbabilistic Scores:")
        if probScoresIn is not None:
            prob_row1 = list(probScoresIn.keys())
        else:
            prob_row1 = list(probScoresOut.keys())
        prob_row1.insert(0, '-')
        if probScoresIn is not None:
            prob_row2 = list(probScoresIn.values())
            prob_row2.insert(0, "InOfDist")
        if probScoresOut is not None:
            prob_row3 = list(probScoresOut.values())
            prob_row3.insert(0, "OutOfDist")
        if probScoresIn is not None:
            if probScoresOut is not None:
                prob_table = [prob_row1, prob_row2, prob_row3]
            else:
                prob_table = [prob_row1, prob_row2]
        else:
            prob_table = [prob_row1, prob_row3]
        print(tabulate(prob_table, headers='firstrow', tablefmt='fancy_grid'))
    
def plot_scores(
    scores_det=None,
    scores_probIn=None,
    scores_probOut=None,
    train_losses=None,
    batchSize=None,
    trainSetSize=None,
    save_folder=None,
    initial_eval=True,
    lastEpoch=0,
):
    """
    Create plots of evolution of given scores over epochs.
    Separate subplots for deterministic and probabilistic errors.
    scores_det and scores_probIn/Out: dictionary, train_losses: list. If train_losses is given then batch size and set size must be given.
    """
    
    # Also put out the raw data
    writeData(scores_det, scores_probIn, scores_probOut, train_losses, save_folder, initial_eval)
    
    if scores_det[0] is None and (scores_probIn[0] is not None or scores_probOut is not None):
        if len(train_losses) == 0:
            fig, axs = plt.subplots(1)
            fig.tight_layout(pad=3.0)
            fig.set_size_inches(8,7)
            if scores_probIn is not None:
                probMatIn = dictToArray(scores_prob=scores_probIn)
            if scores_probOut is not None:
                probMatOut = dictToArray(scores_prob=scores_probOut)
            if not initial_eval:
                if scores_probIn is not None:
                    pad = np.full(probMatIn.shape[0], nP.NaN)
                    probMatIn = np.column_stack((pad, probMatIn))
                if scores_probOut is not None:
                    pad = np.full(probMatOut.shape[0], nP.NaN)
                    probMatOut = np.column_stack((pad, probMatOut))
            if scores_probIn is not None:
                numProbLosses = probMatIn.shape[0]
                numProbEpochs = probMatIn.shape[1]
            else:
                numProbLosses = probMatOut.shape[0]
                numProbEpochs = probMatOut.shape[1]
            for scoreIdx in range(numProbLosses):
                x = np.arange(lastEpoch, lastEpoch+numProbEpochs)
                if scores_probIn is not None:
                    axs.plot(x, probMatIn[scoreIdx,:], label=list(scores_probIn[0].keys())[scoreIdx])
                if scores_probOut is not None:
                    axs.plot(x, probMatOut[scoreIdx,:], label=list(scores_probOut[0].keys())[scoreIdx], linestyle='dashed')
            axs.set_yscale('log')
            axs.grid(True)
            axs.set(xlabel='Epoch', ylabel='Error')
            axs.set_xticks(range(lastEpoch, lastEpoch+numProbEpochs))
            axs.legend()
            axs.set_title('Probabilistic errors')
        else:
            fig, axs = plt.subplots(2)
            fig.tight_layout(pad=3.0)
            fig.set_size_inches(8,7)
            if scores_probIn is not None:
                probMatIn = dictToArray(scores_prob=scores_probIn)
            if scores_probOut is not None:
                probMatOut = dictToArray(scores_prob=scores_probOut)
            if not initial_eval:
                if scores_probIn is not None:
                    pad = np.full(probMatIn.shape[0], nP.NaN)
                    probMatIn = np.column_stack((pad, probMatIn))
                if scores_probOut is not None:
                    pad = np.full(probMatOut.shape[0], nP.NaN)
                    probMatOut = np.column_stack((pad, probMatOut))
            if scores_probIn is not None:
                numProbLosses = probMatIn.shape[0]
                numProbEpochs = probMatIn.shape[1]
            else:
                numProbLosses = probMatOut.shape[0]
                numProbEpochs = probMatOut.shape[1]
            for scoreIdx in range(numProbLosses):
                x = np.arange(lastEpoch, lastEpoch+numProbEpochs)
                if scores_probIn is not None:
                    axs[1].plot(x, probMatIn[scoreIdx,:], label=list(scores_probIn[0].keys())[scoreIdx])
                if scores_probOut is not None:
                    axs[1].plot(x, probMatOut[scoreIdx,:], label=list(scores_probOut[0].keys())[scoreIdx], linestyle='dashed')
            axs[1].set_yscale('log')
            axs[1].grid(True)
            axs[1].set(xlabel='Epoch', ylabel='Error')
            axs[1].set_xticks(range(lastEpoch, lastEpoch+numProbEpochs))
            axs[1].legend()
            axs[1].set_title('Probabilistic errors')
        
    elif scores_det[0] is not None and (scores_probIn[0] is None and scores_probOut is None):
        fig, axs = plt.subplots(1)
        fig.tight_layout(pad=3.0)
        fig.set_size_inches(8,7)
        detMat = dictToArray(scores_det=scores_det)
        if not initial_eval:
            pad = np.full(detMat.shape[0], nP.NaN)
            detMat = np.column_stack((pad, detMat))
        for scoreIdx in range(detMat.shape[0]):
            x = np.arange(lastEpoch, lastEpoch+detMat.shape[1])
            axs.plot(x, detMat[scoreIdx,:], label=list(scores_det[0].keys())[scoreIdx])
        axs.set_yscale('log')
        axs.grid(True)
        axs.set(xlabel='Epoch', ylabel='Error')
        axs.set_xticks(range(lastEpoch, lastEpoch+detMat.shape[1]))
        axs.legend()
        axs.set_title('Deterministic errors')
        
    elif scores_det[0] is not None and (scores_probIn[0] is not None or scores_probOut is not None):
        fig, axs = plt.subplots(2)
        fig.tight_layout(pad=3.0)
        fig.set_size_inches(8,7)
        detMat = dictToArray(scores_det=scores_det)
        if scores_probIn is not None:
            probMatIn = dictToArray(scores_prob=scores_probIn)
        if scores_probOut is not None:
            probMatOut = dictToArray(scores_prob=scores_probOut)
        if not initial_eval:
            if scores_probIn is not None:
                pad = np.full(probMatIn.shape[0], nP.NaN)
                probMatIn = np.column_stack((pad, probMatIn))
            if scores_probOut is not None:
                pad = np.full(probMatOut.shape[0], nP.NaN)
                probMatOut = np.column_stack((pad, probMatOut))
        if not initial_eval:
                pad = np.full(detMat.shape[0], nP.NaN)
                detMat = np.column_stack((pad, detMat))
        if scores_probIn is not None:
            numProbLosses = probMatIn.shape[0]
            numProbEpochs = probMatIn.shape[1]
        else:
            numProbLosses = probMatOut.shape[0]
            numProbEpochs = probMatOut.shape[1]
        for scoreIdx in range(detMat.shape[0]):
            x = np.arange(lastEpoch, lastEpoch+detMat.shape[1])
            axs[0].plot(x, detMat[scoreIdx,:], label=list(scores_det[0].keys())[scoreIdx])
        for scoreIdx in range(numProbLosses):
            x = np.arange(lastEpoch, lastEpoch+numProbEpochs)
            if scores_probIn is not None:
                axs[1].plot(x, probMatIn[scoreIdx,:], label=list(scores_probIn[0].keys())[scoreIdx])
            if scores_probOut is not None:
                axs[1].plot(x, probMatOut[scoreIdx,:], label=list(scores_probOut[0].keys())[scoreIdx], linestyle='dashed')
        for ax in axs.flat:
            ax.set_yscale('log')
            ax.grid(True)
            ax.set(xlabel='Epoch', ylabel='Error')
            ax.set_xticks(range(lastEpoch, lastEpoch+detMat.shape[1]))
            ax.legend()
        axs[0].set_title('Deterministic errors')
        axs[1].set_title('Probabilistic errors')
    
    if len(train_losses) != 0:
        if scores_det[0] is not None and (scores_probIn[0] is None and scores_probOut is None):
            x_loss = np.linspace(0, len(scores_det)-1, len(train_losses))
            axs.plot(x_loss, train_losses, label='Training loss')
            axs.legend()
        elif scores_det[0] is None and (scores_probIn[0] is None and scores_probOut is None):
            fig, axs = plt.subplots(1)
            fig.tight_layout(pad=3.0)
            fig.set_size_inches(8,7)
            x_loss = np.linspace(0, len(train_losses), len(train_losses))
            axs.plot(x_loss, train_losses, label='Training loss')
            axs.set_yscale('log')
            axs.grid(True)
            axs.set(xlabel='Batch', ylabel='Loss')
            axs.set_title('Trainig loss')
            axs.legend()
        else:
            x_loss = np.linspace(0, len(scores_det)-1, len(train_losses))
            axs[0].plot(x_loss, train_losses, label='Training loss')
            axs[0].set_yscale('log')
            axs[0].grid(True)
            axs[0].legend()
    
    if save_folder is None:
        plt.savefig("train_errors.png")
    else:
        plt.savefig(os.path.join(save_folder, "train_errors.png"))
    
    
def dictToArray(
    scores_det=None,
    scores_prob=None
):
    """
    Helper function to turn a dictionary of scores into an array.
    Separate arrays for deterministic and probabilistic.
    array.size = [num of epochs, num of different scores] (I think...)
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

def writeData(scores_det, scores_probIn, scores_probOut, train_losses, save_folder, initial_eval):
    """
    Prints scores.
    """
    if save_folder is None:
        file = open('train_errors_data.txt', 'w')
    else:
        file = open(os.path.join(save_folder, 'train_errors_data.txt'), 'w')
        
    if initial_eval:
        file.write('Initial eval: True\n')
    else:
        file.write('Initial eval: False\n')
    
    if scores_det[0] is not None:
        file.write('Deterministic scores\n')
        for loss in scores_det[0]:
            file.write(loss+' ')
        file.write('\n')

        for dict in scores_det:
            file.write('\n')
            for loss in dict:
                file.write(f'{dict[loss]} ')
            
        file.write('\n')
        
    if scores_probIn[0] is not None and scores_probOut is not None:
        file.write('\nProbabilistic scores\n')
        for loss in scores_probIn[0]:
            file.write(loss+' ')
        file.write('\n')

        file.write('InDist\n')
        for dict in scores_probIn:
            file.write('\n')
            for loss in dict:
                file.write(f'{dict[loss]} ')

        file.write('\nInDist\n')
        for dict in scores_probOut:
            file.write('\n')
            for loss in dict:
                file.write(f'{dict[loss]} ')
        
        file.write('\n')

    elif scores_probIn[0] is not None and scores_probOut is None:
        file.write('\nProbabilistic scores\n')
        for loss in scores_probIn[0]:
            file.write(loss+' ')
        file.write('\n')

        file.write('InDist\n')
        for dict in scores_probIn:
            file.write('\n')
            for loss in dict:
                file.write(f'{dict[loss]} ')

        file.write('\n')

    elif scores_probIn[0] is None and scores_probOut is not None:
        file.write('\nProbabilistic scores\n')
        for loss in scores_probOut[0]:
            file.write(loss+' ')
        file.write('\n')

        file.write('InDist\n')
        for dict in scores_probOut:
            file.write('\n')
            for loss in dict:
                file.write(f'{dict[loss]} ')

        file.write('\n')
        
    if len(train_losses) != 0:
        file.write('\nTraining loss\n')
        for loss in train_losses:
            file.write(f'\n{loss}')
            
    file.close()