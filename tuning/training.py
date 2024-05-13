"""
"""
import torch
import sys
import os
import time
#import zarr
from ray import train
from ray.train import Checkpoint

from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_shear_flow
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss, compute_probabilistic_scores, compute_deterministic_scores, print_scores, hacky_crps, plot_scores, lognormal_crps, ensemble_crps, rbf, mmd

def training(config):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print('Warning: No GPU available! Training on CPU')
    else:
        print(f'GPU in use: {torch.cuda.get_device_name()}')
        
    # Load the Navier--Stokes dataset
    batch_size = 32
    n_train = 10
    n_epochs = 2
    predicted_t = 10
    n_tests = 10
    train_loader, test_loaders, ensemble_loader, data_processor = load_shear_flow(
            n_train=n_train,             # 40_000
            batch_size=batch_size, 
            train_resolution=128,
            test_resolutions=[128],  # [64,128], 
            n_tests=[n_tests],           # [10_000, 10_000],
            test_batch_sizes=[32],  # [32, 32],
            positional_encoding=True,
            T=predicted_t
    )
    data_processor = data_processor.to(device)
    
    # Create a tensorized FNO model
    model = TFNO(
                n_modes=config['n_modes'],
                in_channels=4,
                out_channels=2,
                hidden_channels=config['hidden_channels'],
                n_layers=config['n_layers'],
                projection_channels=64, 
                factorization='tucker', 
                rank=0.42)
    # in_channels = 2 physical variables + 2 positional encoding
    model = model.to(device)
    
    n_params = count_model_params(model)
    print(f'\Model has {n_params} parameters.')
    sys.stdout.flush()

    #Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
              
    # Create the losses
    reduce_dims = 0
    reductions = 'mean'
    l2loss = LpLoss(d=2, p=2, reduce_dims=reduce_dims, reductions=reductions)
    h1loss = H1Loss(d=2, reduce_dims=reduce_dims, reductions=reductions)

    train_loss = h1loss
    eval_losses = {'l2': l2loss, 'h1': h1loss}

    # Probabilistic score metrics
    probab_scores = None
    #crps = lognormal_crps(member_dim=0, reduce_dims=None)
    ensembleCrps = ensemble_crps(member_dim=0, reduce_dims=None)
    
    median_sigma = torch.median(ensemble_loader.dataset[0]['y'][:,:,:])
    gauss_kernel = rbf(median_sigma)
    maxMeanDiscr = mmd(gauss_kernel, member_dim=0, reduce_dims=None)

    probab_scores = {'mmd': maxMeanDiscr, 'ensemble_crps': ensembleCrps}
    
    # Create the trainer
    trainer = Trainer(model=model, n_epochs=1,
                      device=device,
                      data_processor=data_processor,
                      wandb_log=False,
                      log_test_interval=1,
                      use_distributed=False,
                      verbose=True)


    # %%
    # Actually train the model
    n_modes = config['n_modes']
    hidden_channels = config['hidden_channels']
    n_layers = config['n_layers']
    run_directory = '/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/tuning/runs/'
    run_directory = os.path.join(run_directory, f'n_modes={n_modes}_hidden_channels={hidden_channels}_n_layers={n_layers}')
    os.mkdir(run_directory)
    train_folder = os.path.join(run_directory, 'training')
    os.mkdir(train_folder)
    start = time.time()
    trainErrors_det = []
    trainErrors_prob = []
    train_losses = []
    for i in range(n_epochs):
        initial_eval = False
        if i == 0:
              initial_eval = True
        NewtrainErrors_det, NewtrainErrors_prob, Newtrain_losses = trainer.train(train_loader=train_loader,
                                                                      test_loaders=test_loaders,
                                                                      optimizer=optimizer,
                                                                      scheduler=scheduler, 
                                                                      regularizer=False,
                                                                      save_folder=train_folder,#ensemble_loader=ensemble_loader,
                                                                      training_loss=train_loss,
                                                                      eval_losses=eval_losses,
                                                                      prob_losses=probab_scores,
                                                                      loss_reductions=reductions,
                                                                      initial_eval=initial_eval,
                                                                      checkpoint=False)
        #model.save_checkpoint(save_folder=train_folder, save_name='example_fno_shear')
        if initial_eval:
            assert len(NewtrainErrors_det) == 2, 'Wrong train error list length: {len(NewtrainErrors_det)} instead of 2.'
        else:
            assert len(NewtrainErrors_det) == 1, 'Wrong train error list length: {len(NewtrainErrors_det)} instead of 1.'
        trainErrors_det += NewtrainErrors_det
        trainErrors_prob += NewtrainErrors_prob
        train_losses += Newtrain_losses
              
        checkpoint = None
        epoch_dir = os.path.join(run_directory, f"epoch={i}")
        os.mkdir(epoch_dir)
        if (i + 1) % 5 == 0:
            # This saves the model to the trial directory
            torch.save(
                model.state_dict(),
                os.path.join(epoch_dir, "model.pth")
            )
            checkpoint = Checkpoint.from_directory(epoch_dir)

        # Send the current training result back to Tune
        train.report({"h1": trainErrors_det[i+1]['128_h1']}, checkpoint=checkpoint)
    
    end = time.time()
    print(f'Training took {end-start} s.')

    plot_scores(trainErrors_det, trainErrors_prob, train_losses, batch_size, n_train, save_folder=run_directory)
    
    