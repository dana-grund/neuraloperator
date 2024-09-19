import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math

from neuralop.models import TFNO
from neuralop.datasets import load_shear_flow, plot_shear_flow_test

from neuralop import LpLoss, H1Loss, MedianAbsoluteLoss, gaussian_crps, compute_probabilistic_scores, compute_deterministic_scores, print_scores

"""
Script for plotting energy spactra of the flow.
Contains functions for different forms of plots (1D, 2D, ensemble, etc.).
Running this script plots 2D and 1D spectra of predicted flows of the ensemble dataset.
"""

def forwardPass(model, loader, num_batches, processor):
    """
    """
    outStack = None
    labelStack = None
    i = 0
    for batch in loader:
        if i < num_batches:
            outputs = model(**processor.preprocess(batch))
            outputs, inputs = processor.postprocess(outputs, batch)
            if i == 0:
                outStack = outputs.detach().numpy()
                labelStack = batch['y'].detach().numpy()
            else:
                outStack = np.concatenate((outStack, outputs.detach().numpy()), axis=0)
                labelStack = np.concatenate((labelStack, batch['y'].detach().numpy()), axis=0)
            i += 1
        else:
            break
    return outStack, labelStack


def transformTotEnergie(sqVelocities):
    energies = 0.5 * (sqVelocities[:,0,:,:] + sqVelocities[:,1,:,:])
    transfEnergies = np.fft.fft2(energies)
    transfEnergies = np.fft.fftshift(transfEnergies)
    transfEnergies = np.abs(transfEnergies)
    return transfEnergies
    
    
def transformOneCompEnergy(sqVelocities, direction):
    """
    direction = 0 -> x (or u in velocity)
    direction = 1 -> y (or v in velocity)
    """
    energies = 0.5 * sqVelocities[:,direction,:,:]
    transfEnergies = np.fft.fft2(energies)
    transfEnergies = np.fft.fftshift(transfEnergies)
    transfEnergies = np.abs(transfEnergies)
    print(transfEnergies.shape)
    return transfEnergies

def plot2DSpectrum(transforms, frequencies, component, labelTransf=None, regularData=False):
    fig, axs = plt.subplots(2,2, figsize=(8.5, 10))
    fig.set_figheight(7)
    
    # Compute global min and max
    meansPred = np.mean(transforms, axis=0)
    meansLab = np.mean(labelTransf, axis=0)
    allData = [meansPred, meansLab, transforms[0,:,:], labelTransf[0,:,:]]
    minGlob = np.min([np.min(data) for data in allData])
    maxGlob = np.max([np.max(data) for data in allData])
    
    # Predicted Ensemble mean
    cax = axs[0,0].imshow(meansPred, extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
    fig.colorbar(cax, ax=axs[0,0], orientation='vertical')
    axs[0,0].set(ylabel='y-Frequency')
    axs[0,0].set_title(f'Predicted Ensemble Mean')
    
    # Label Ensemble mean
    cax = axs[0,1].imshow(meansLab, extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
    fig.colorbar(cax, ax=axs[0,1], orientation='vertical', label="Energy")
    axs[0,1].set_title(f'Ground Truth Ensemble Mean')
    
    # First member prediction
    cax = axs[1,0].imshow(transforms[0,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
    fig.colorbar(cax, ax=axs[1,0], orientation='vertical')
    axs[1,0].set(xlabel='x-Frequency', ylabel='y-Frequency')
    axs[1,0].set_title(f'Single Predicted Member')
    
    # First label
    if labelTransf is not None:
        cax = axs[1,1].imshow(labelTransf[0,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
        fig.colorbar(cax, ax=axs[1,1], orientation='vertical', label="Energy")
        axs[1,1].set(xlabel='x-Frequency')
        axs[1,1].set_title(f'Single Ground Truth Member')
    
    fig.subplots_adjust(hspace=0.4, wspace=0.2, top=0.9)
    fig.suptitle("2D-Spectra - Total Kinetic Energy", fontsize=16, x=0.5, y=0.98)
    
    plt.tight_layout()
    if not regularData:
        plt.savefig(f"correctedEns_model/{component}Kinetic2D.png")
    else:
        plt.savefig(f"correctedEns_model/regularData_{component}Kinetic2D.png")
    
    # Differences
    l1loss = LpLoss(d=2, p=1, reduce_dims=0, reductions='mean')
    diffFirsts = l1loss(torch.from_numpy(labelTransf[0,:,:]), torch.from_numpy(transforms[0,:,:]))
    diffMeans = l1loss(torch.from_numpy(meansLab), torch.from_numpy(meansPred))
    
    return diffMeans, diffFirsts


def plotMulti2DSpectrum(transforms, frequencies, component, labelTransf=None, regularData=False):
    fig, axs = plt.subplots(2,2, figsize=(8.5, 10))
    fig.set_figheight(7)
    
    # Compute global min and max
    # meansPred = np.mean(transforms, axis=0)
    # meansLab = np.mean(labelTransf, axis=0)
    # allData = [meansPred, meansLab, transforms[0,:,:], labelTransf[0,:,:]]
    # minGlob = np.min([np.min(data) for data in allData])
    # maxGlob = np.max([np.max(data) for data in allData])
    
    # Predicted Ensemble mean
    cax = axs[0,0].imshow(transforms[0,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
    fig.colorbar(cax, ax=axs[0,0], orientation='vertical')
    axs[0,0].set(ylabel='y-Frequency')
    axs[0,0].set_title(f'Sample 1, Prediction')
    
    # Label Ensemble mean
    cax = axs[0,1].imshow(labelTransf[0,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
    fig.colorbar(cax, ax=axs[0,1], orientation='vertical', label="Energy")
    axs[0,1].set_title(f'Sample 1, Ground Truth')
    
    # First member prediction
    cax = axs[1,0].imshow(transforms[1,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
    fig.colorbar(cax, ax=axs[1,0], orientation='vertical')
    axs[1,0].set(xlabel='x-Frequency', ylabel='y-Frequency')
    axs[1,0].set_title(f'Sample 2, Prediction')
    
    # First label
    if labelTransf is not None:
        cax = axs[1,1].imshow(labelTransf[0,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
        fig.colorbar(cax, ax=axs[1,1], orientation='vertical', label="Energy")
        axs[1,1].set(xlabel='x-Frequency')
        axs[1,1].set_title(f'Sample 2, Ground Truth')
    
    fig.subplots_adjust(hspace=0.4, wspace=0.2, top=0.9)
    fig.suptitle("2D-Spectra - Total Kinetic Energy", fontsize=16, x=0.5, y=0.98)
    
    plt.tight_layout()
    if not regularData:
        plt.savefig(f"correctedEns_model/multi{component}Kinetic2D.png")
    else:
        plt.savefig(f"correctedEns_model/regularData_multi{component}Kinetic2D.png")



def plot1DSpectrum(transforms, frequencies, component, labelTransf=None):
    # Determin radi step
    #maxFreq = np.max(np.abs(frequencies))
    #maxRad = maxFreq * math.sqrt(2)
    #tol = 1e-5
    #radi = np.linspace(-tol, maxRad + tol, frequencies.size+1)
    
    fig, axs = plt.subplots(3, figsize=(10, 10))
    
    # Mean over ensemble
    meansPred = np.mean(transforms, axis=0)
    
    # Calculate all radi
    radi = np.empty((meansPred.size))
    for i in range(frequencies.size):
        for j in range(frequencies.size):
            radi[i*frequencies.size + j] = math.sqrt(frequencies[i]**2 + frequencies[j]**2)
            
    numBins = frequencies.size
    
    # All seperate predicted members
    for i in range(transforms.shape[0]):
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=transforms[i].flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs[0].loglog(binCentersMember, histMember, alpha=0.1, color='grey')
    
    # Create histogram on radi with coefficients as weights
    hist, binEdges = np.histogram(radi, bins=numBins, density=False, weights=meansPred.flatten())
    binCenters = (binEdges[:-1]+binEdges[1:]) / 2.0
    axs[0].loglog(binCenters, hist, label="Ensemble mean")
    axs[0].set_title(f'Predicted {component} Kinetic Energy per Mass')
    # Plot with both means
    axs[2].loglog(binCenters, hist, label="Prediction mean")
    
    # All seperate label members
    for i in range(labelTransf.shape[0]):
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=labelTransf[i].flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs[1].loglog(binCentersMember, histMember, alpha=0.1, color='grey')
        
    # Label mean energies
    meansLab = np.mean(labelTransf, axis=0)
    hist, binEdges = np.histogram(radi, bins=numBins, density=False, weights=meansLab.flatten())
    binCenters = (binEdges[:-1]+binEdges[1:]) / 2.0
    axs[1].loglog(binCenters, hist, label="Ensemble mean")
    axs[1].set_title(f'Label {component} Kinetic Energy per Mass')
    # Plot with both means
    axs[2].loglog(binCenters, hist, label="Label mean")
    
    axs[2].set_title(f'Mean {component} Kinetic Energies per Mass')
    for ax in axs.flat:
        ax.set(xlabel='Frequencie', ylabel='Energy')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"correctedEns_model/{component}Kinetic1D.png")
    

# Plot mean spectrum of all ensembles !! Needs all data. Not only one ensemble !!
def plotMulti2DSpectrum_ensembles(transforms, frequencies, component, labelTransf=None, regularData=False):
    # plotting
    fig, axs = plt.subplots(5,2, figsize=(20, 20))
    fig.set_figwidth(9)
    
    # Compute global min and max
    allData = [transforms[0,:,:], transforms[1,:,:], transforms[2,:,:], labelTransf[0,:,:], labelTransf[1,:,:], labelTransf[2,:,:]]
    minGlob = np.min([np.min(data) for data in allData])
    maxGlob = np.max([np.max(data) for data in allData])
    
    for i in range(10):
        col = i % 2
        row = i // 2

        cax = axs[row, col].imshow(np.mean(transforms[i*100:(i+1)*100,:,:], axis=0), extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
        fig.colorbar(cax, ax=axs[row,col], fraction=0.046, pad=0.04)

        if col == 0:
            axs[row,col].set(xlabel='x-Frequency', ylabel='y-Frequency')
        else:
            axs[row,col].set(xlabel='x-Frequency')
        axs[row,col].set_title(f'Ensemble {i+1}')

    fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.9)
    fig.suptitle("Predicted Ensemble Means Spectra - Total Kinetic Energy", fontsize=16, x=0.43, y=0.92)  # Raise the title

    #cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

    # ticks = np.arange(0, 128, 20)
    # for ax in axs.flat:
    #     ax.set_xticks(ticks)
    #     ax.set_yticks(ticks)

    #plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
    fig.tight_layout(rect=[0, 0, 0.85, 0.93])

    plt.savefig(f"correctedEns_model/predEnsembleMeans_{component}Kinetic2D.png")

    # plotting
    fig, axs = plt.subplots(5,2, figsize=(20, 20))
    fig.set_figwidth(9)
    
    for i in range(10):
        col = i % 2
        row = i // 2

        cax = axs[row, col].imshow(np.mean(labelTransf[i*100:(i+1)*100,:,:], axis=0), extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm())
        fig.colorbar(cax, ax=axs[row,col], fraction=0.046, pad=0.04)

        if col == 0:
            axs[row,col].set(xlabel='x-Frequency', ylabel='y-Frequency')
        else:
            axs[row,col].set(xlabel='x-Frequency')
        axs[row,col].set_title(f'Ensemble {i+1}')

    fig.subplots_adjust(hspace=0.3, wspace=0.4, top=0.9)
    fig.suptitle(f"Label Ensemble Means Spectra - {component} Kinetic Energy", fontsize=16, x=0.43, y=0.92)  # Raise the title

    #cbar_ax = fig.add_axes([0.87, 0.095, 0.03, 0.7])

    # ticks = np.arange(0, 128, 20)
    # for ax in axs.flat:
    #     ax.set_xticks(ticks)
    #     ax.set_yticks(ticks)

    #plt.colorbar(caxList[maxGlobI], cax=cbar_ax)
    fig.tight_layout(rect=[0, 0, 0.85, 0.93])

    plt.savefig(f"correctedEns_model/labelEnsembleMeans_{component}Kinetic2D.png")
    
    

# Same but members from differnt ensembles
def plotMulti2DSpectrumDiffEns(transforms, frequencies, component, labelTransf=None, regularData=False):
    fig, axs = plt.subplots(3,2, figsize=(10, 15))
    
    # Compute global min and max
    allData = [transforms[0,:,:], transforms[1,:,:], transforms[2,:,:], labelTransf[0,:,:], labelTransf[1,:,:], labelTransf[2,:,:]]
    minGlob = np.min([np.min(data) for data in allData])
    maxGlob = np.max([np.max(data) for data in allData])
    
    # First member prediction
    cax = axs[0,0].imshow(transforms[0,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm(vmin=minGlob, vmax=maxGlob))
    fig.colorbar(cax, ax=axs[0,0], orientation='vertical', label="Energy")
    axs[0,0].set(xlabel='x-Frequencie', ylabel='y-Frequencie')
    axs[0,0].set_title(f'Prediction {component} Kinetic Energy per Mass, Ensemble 1')
    
    # First label
    if labelTransf is not None:
        cax = axs[0,1].imshow(labelTransf[0,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm(vmin=minGlob, vmax=maxGlob))
        fig.colorbar(cax, ax=axs[0,1], orientation='vertical', label="Energy")
        axs[0,1].set(xlabel='x-Frequencie', ylabel='y-Frequencie')
        axs[0,1].set_title(f'First Label {component} Kinetic Energy per Mass, Ensemble 1')
    
    # Second member prediction
    cax = axs[1,0].imshow(transforms[1,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm(vmin=minGlob, vmax=maxGlob))
    fig.colorbar(cax, ax=axs[1,0], orientation='vertical', label="Energy")
    axs[1,0].set(xlabel='x-Frequencie', ylabel='y-Frequencie')
    axs[1,0].set_title(f'Second Prediction {component} Kinetic Energy per Mass, Ensembel 2')
    
    # Second label
    if labelTransf is not None:
        cax = axs[1,1].imshow(labelTransf[1,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm(vmin=minGlob, vmax=maxGlob))
        fig.colorbar(cax, ax=axs[1,1], orientation='vertical', label="Energy")
        axs[1,1].set(xlabel='x-Frequencie', ylabel='y-Frequencie')
        axs[1,1].set_title(f'Second Label {component} Kinetic Energy per Mass, Ensembel 2')
        
    # Third member prediction
    cax = axs[2,0].imshow(transforms[2,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm(vmin=minGlob, vmax=maxGlob))
    fig.colorbar(cax, ax=axs[2,0], orientation='vertical', label="Energy")
    axs[2,0].set(xlabel='x-Frequencie', ylabel='y-Frequencie')
    axs[2,0].set_title(f'Third Prediction {component} Kinetic Energy per Mass, Ensembel 3')
    
    # Third label
    if labelTransf is not None:
        cax = axs[2,1].imshow(labelTransf[2,:,:], extent=[frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]], norm=LogNorm(vmin=minGlob, vmax=maxGlob))
        fig.colorbar(cax, ax=axs[2,1], orientation='vertical', label="Energy")
        axs[2,1].set(xlabel='x-Frequencie', ylabel='y-Frequencie')
        axs[2,1].set_title(f'Third Label {component} Kinetic Energy per Mass, Ensembel 3')
    
    plt.tight_layout()
    if not regularData:
        plt.savefig(f"correctedEns_model/multiEns_{component}Kinetic2D.png")
    else:
        plt.savefig(f"correctedEns_model/regularData_multiEns_{component}Kinetic2D.png")


def plotMulti1DSpectrum(transforms, frequencies, component, labelTransf=None, regularData=False):
    # Determin radi step
    #maxFreq = np.max(np.abs(frequencies))
    #maxRad = maxFreq * math.sqrt(2)
    #tol = 1e-5
    #radi = np.linspace(-tol, maxRad + tol, frequencies.size+1)
    
    fig, axs = plt.subplots(2, figsize=(10, 6))
    
    # Mean over ensemble
    meansPred = np.mean(transforms, axis=0)
    
    # Calculate all radi
    radi = np.empty((meansPred.size))
    for i in range(frequencies.size):
        for j in range(frequencies.size):
            radi[i*frequencies.size + j] = math.sqrt(frequencies[i]**2 + frequencies[j]**2)
            
    numBins = frequencies.size
    
    for i in range(2):
        # Predicted
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=transforms[i].flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs[i].loglog(binCentersMember, histMember, label="Prediction", color='blue')
        
        # Label
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=labelTransf[i].flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs[i].loglog(binCentersMember, histMember, label="Ground Truth", color='red')
        
        axs[i].set_title(f"Sample {i+1}")
    
    for ax in axs.flat:
        ax.legend()
    
    axs[0].set(ylabel='Energy')
    axs[1].set(xlabel='Frequency', ylabel='Energy')
    
    fig.suptitle(f"Ensemble Mean Spectrum - {component} Kinetic Energy", fontsize=16)

    plt.tight_layout()
    if regularData:
        plt.savefig(f"correctedEns_model/regularData_multi2_{component}Kinetic1D.png")
    else:
        plt.savefig(f"correctedEns_model/multi2_{component}Kinetic1D.png")


# Plot mean spectrum of all ensembles !! Needs all data. Not only one ensemble !!
def plot1DSpectrum_ensembles(transforms, frequencies, component, labelTransf=None, regularData=False):
    # Determin radi step
    #maxFreq = np.max(np.abs(frequencies))
    #maxRad = maxFreq * math.sqrt(2)
    #tol = 1e-5
    #radi = np.linspace(-tol, maxRad + tol, frequencies.size+1)
    
    fig, axs = plt.subplots(3, figsize=(10, 10))
    
    # Mean over ensemble
    meansPred = np.mean(transforms, axis=0)
    
    # Calculate all radi
    radi = np.empty((meansPred.size))
    for i in range(frequencies.size):
        for j in range(frequencies.size):
            radi[i*frequencies.size + j] = math.sqrt(frequencies[i]**2 + frequencies[j]**2)
            
    numBins = frequencies.size
    
    for i in range(3):
        # All seperate predicted members
        for j in range(100):
            histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=transforms[i*100+j].flatten())
            binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
            axs[i].loglog(binCentersMember, histMember, alpha=0.25, color='cyan')
            histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=labelTransf[i*100+j].flatten())
            binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
            axs[i].loglog(binCentersMember, histMember, alpha=0.25, color='orange')
        
        ensMean = np.mean(transforms[i*100:(i+1)*100], axis=0)
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=ensMean.flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs[i].loglog(binCentersMember, histMember, label="Predicted Mean", color='blue')
        
        ensMean = np.mean(labelTransf[i*100:(i+1)*100], axis=0)
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=ensMean.flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs[i].loglog(binCentersMember, histMember, label="Label Mean", color='red')
        
        axs[i].set_title(f"Ensemble {i+1}")
    
    for ax in axs.flat:
        ax.set(xlabel='Frequency', ylabel='Energy')
        ax.legend()
    
    fig.suptitle(f"Ensemble Means Spectra - {component} Kinetic Energy", fontsize=16)
    
    plt.tight_layout()
    if regularData:
        plt.savefig(f"correctedEns_model/regularData_ensembleMeans_{component}Kinetic1D.png")
    else:
        plt.savefig(f"correctedEns_model/ensembleMeans1to3_{component}Kinetic1D.png")

    fig, axs = plt.subplots(3, figsize=(10, 10))

    for i in range(3, 6):
        # All seperate predicted members
        for j in range(100):
            histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=transforms[i*100+j].flatten())
            binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
            axs[i-3].loglog(binCentersMember, histMember, alpha=0.25, color='cyan')
            histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=labelTransf[i*100+j].flatten())
            binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
            axs[i-3].loglog(binCentersMember, histMember, alpha=0.25, color='orange')

        # Predicted
        ensMean = np.mean(transforms[i*100:(i+1)*100], axis=0)
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=ensMean.flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs[i-3].loglog(binCentersMember, histMember, label="Predicted Mean", color='blue')
        
        # Label
        ensMean = np.mean(labelTransf[i*100:(i+1)*100], axis=0)
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=ensMean.flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs[i-3].loglog(binCentersMember, histMember, label="Label Mean", color='red')
        
        axs[i-3].set_title(f"Ensemble {i+1}")
    
    for ax in axs.flat:
        ax.set(xlabel='Frequency', ylabel='Energy')
        ax.legend()
    
    fig.suptitle(f"Ensemble Means Spectra - {component} Kinetic Energy", fontsize=16)
    
    plt.tight_layout()
    if regularData:
        plt.savefig(f"correctedEns_model/regularData_ensembleMeans_{component}Kinetic1D.png")
    else:
        plt.savefig(f"correctedEns_model/ensembleMeans3to6_{component}Kinetic1D.png")
    
# Plot mean spectrum of frist ensemble
def plot1DSpectrum_ensemble1(transforms, frequencies, component, labelTransf=None, regularData=False):
    # Determin radi step
    #maxFreq = np.max(np.abs(frequencies))
    #maxRad = maxFreq * math.sqrt(2)
    #tol = 1e-5
    #radi = np.linspace(-tol, maxRad + tol, frequencies.size+1)
    
    fig, axs = plt.subplots(1, figsize=(10, 3.5))
    
    # Mean over ensemble
    meansPred = np.mean(transforms, axis=0)
    
    # Calculate all radi
    radi = np.empty((meansPred.size))
    for i in range(frequencies.size):
        for j in range(frequencies.size):
            radi[i*frequencies.size + j] = math.sqrt(frequencies[i]**2 + frequencies[j]**2)
            
    numBins = frequencies.size
    
    i = 0
    for j in range(100):
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=transforms[i*100+j].flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs.loglog(binCentersMember, histMember, alpha=0.25, color='cyan')
        histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=labelTransf[i*100+j].flatten())
        binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
        axs.loglog(binCentersMember, histMember, alpha=0.25, color='orange')
    
    ensMean = np.mean(transforms[i*100:(i+1)*100], axis=0)
    histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=ensMean.flatten())
    binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
    axs.loglog(binCentersMember, histMember, label="Predicted Mean", color='blue')
    
    ensMean = np.mean(labelTransf[i*100:(i+1)*100], axis=0)
    histMember, binEdgesMember = np.histogram(radi, bins=numBins, density=False, weights=ensMean.flatten())
    binCentersMember = (binEdgesMember[:-1]+binEdgesMember[1:]) / 2.0
    axs.loglog(binCentersMember, histMember, label="Ground Truth Mean", color='red')
    
    axs.set(xlabel='Frequency', ylabel='Energy')
    axs.legend()
    
    fig.suptitle(f"Ensemble Mean Spectrum - {component} Kinetic Energy", fontsize=16)
    
    plt.tight_layout()
    if regularData:
        plt.savefig(f"correctedEns_model/regularData_ensemble1Mean_{component}Kinetic1D.png")
    else:
        plt.savefig(f"correctedEns_model/ensembleMean1_{component}Kinetic1D.png")

    
    
    
# Load datasets
batch_size = 25
n_train = 40000
n_epochs = 5
predicted_t = 10
n_tests = 300
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
#data_processor = data_processor.to(device)

# Load model for forward pass
folder = "/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/energy_plotting/correctedEns_model/"
name = "fno_shear_n_train=40000_epoch=5_correctedEns_cpu"
model = TFNO.from_checkpoint(save_folder=folder, save_name=name)
model.eval()
print("Model loaded")
# model.to(device)

# Forward pass of first ensemble
num_batches = 100 / batch_size
print(f"Num. batches: {num_batches}")
num_batches = int(num_batches)
velocities, labelVels = forwardPass(model, ensemble_loaders[0], num_batches, data_processor)
print("Pass done")
print(velocities.shape)
#print(velocities)

# Debug eval
#reduce_dims = 0
#reductions = 'mean'
#h1loss = H1Loss(d=2, reduce_dims=reduce_dims, reductions=reductions)
#eval_losses = {'h1': h1loss}
#test_db = test_loaders[128].dataset
#absScores, relScores = compute_deterministic_scores(
#    test_db,
#    model,
#    data_processor,
#    eval_losses
#)
#print_scores(scores_abs=absScores, scores_rel=relScores, reductions=reductions)

# Compute squares/energies
sqVelocities = np.square(velocities)
sqLabelVels = np.square(labelVels)

# Get fourier transforms for total energy
transforms = transformTotEnergie(sqVelocities)
labelTransf = transformTotEnergie(sqLabelVels)

# Get sample frequencies
frequencies = np.fft.fftfreq(transforms[0].shape[1])
frequencies = np.fft.fftshift(frequencies)

# Plot
diffMeans, diffFirsts = plot2DSpectrum(transforms, frequencies, 'Total', labelTransf=labelTransf)
print(f'\nRelative Diff total 2D means: {diffMeans}')
print(f'Relative Diff total 2D firsts: {diffFirsts}\n')

plot1DSpectrum(transforms, frequencies, 'Total', labelTransf=labelTransf)

plotMulti1DSpectrum(transforms, frequencies, 'Total', labelTransf=labelTransf)


# # Get fourier transform for energy in x direction
# direction = 0
# UTransforms = transformOneCompEnergy(sqVelocities, direction)
# ULabelTransf = transformOneCompEnergy(sqLabelVels, direction)

# # Get sample frequencies
# frequencies = np.fft.fftfreq(UTransforms[0].shape[1])
# frequencies = np.fft.fftshift(frequencies)

# # Plot 
# diffMeans, diffFirsts = plot2DSpectrum(UTransforms, frequencies, 'XComponent', labelTransf=ULabelTransf)
# print(f'Relative Diff x-component 2D means: {diffMeans}')
# print(f'Relative Diff x-component 2D firsts: {diffFirsts}\n')

# plot1DSpectrum(UTransforms, frequencies, 'XComponent', labelTransf=ULabelTransf)

# plotMulti1DSpectrum(UTransforms, frequencies, 'XComponent', labelTransf=ULabelTransf)


# # Get fourier transform for energy in y direction
# direction = 1
# VTransforms = transformOneCompEnergy(sqVelocities, direction)
# VLabelTransf = transformOneCompEnergy(sqLabelVels, direction)

# # Get sample frequencies
# frequencies = np.fft.fftfreq(VTransforms[0].shape[1])
# frequencies = np.fft.fftshift(frequencies)

# # Plot 
# diffMeans, diffFirsts = plot2DSpectrum(VTransforms, frequencies, 'YComponent', labelTransf=VLabelTransf)
# print(f'Relative Diff y-component 2D means: {diffMeans}')
# print(f'Relative Diff y-component 2D firsts: {diffFirsts}\n')

# plot1DSpectrum(VTransforms, frequencies, 'YComponent', labelTransf=VLabelTransf)

# plotMulti1DSpectrum(VTransforms, frequencies, 'YComponent', labelTransf=VLabelTransf)


