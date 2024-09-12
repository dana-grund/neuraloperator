import torch
import os
import ray
import time
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt

from training import training

"""
Tuning script for the hyperparameter tuning.
To run this script, create an EMPTY directory 'runs' at the same location as this file.
Tuned parameters:
    number of Fourier modes
    number of hidden channels
    number of Fourier layers
"""

start = time.time()

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)
    
print('initialising')
# Initialize tuning with 3 cpus and 2 gpus. -> adjust to your specifications
ray.init(num_cpus=3, num_gpus=2)
print('initialized')

# Grid search for all hyperparameters
search_space = {
    'n_modes': tune.grid_search([(16,16), (32,32), (64,64)]),
    'hidden_channels': tune.grid_search([16, 32, 64]),
    'n_layers': tune.grid_search([4, 8])
}

# Each training uses 1 cpu and 1 gpu. ASHA Scheduler for early stopping depending on h1 loss.
tuner = tune.Tuner(
    tune.with_resources(training, resources={"cpu": 1, "gpu": 1}),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        scheduler=ASHAScheduler(metric="h1", mode="min"),
    )
)

# Perform the tuning
print('Fitting now')
results = tuner.fit()
print('Fitting finished')

dfs = {result.path: result.metrics_dataframe for result in results}

# Plot results
fig, ax = plt.subplots()

for df in dfs.values():
    print(df)
    if not df.empty:
        label = f"Modes{df['config/n_modes'][0]}, Channels{df['config/hidden_channels'][0]}, Layers{df['config/n_layers'][0]}"
        df['h1'].plot(ax=ax, label=label)

ax.set(xlabel='Epoch', ylabel='Error')
ax.set_xticks(range(1, len(df)+1))
ax.legend()
ax.set_title('H1 errors')

plt.savefig("combined_plots.png", bbox_inches='tight')
plt.close(fig)

end = time.time()
print(f'Tuning took {end-start} s.')

# Report path of best performing model
bestModelDir = results.get_best_result('h1', mode='min').path
print(f'\nBest performing model is saved at {bestModelDir}')

ray.shutdown()