import torch
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt

from training import training

"""
To run this script, create an EMPTY directory 'runs' at the same location as this file.
"""

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)
    
print('initialising')
ray.init(num_cpus=1) #num_gpus=3
print('initialized')


search_space = {
    #'n_modes': tune.grid_search([(16,16), (32,32), (64,64)]),
    #'hidden_channels': tune.grid_search([16, 32, 64]),
    #'n_layers': tune.grid_search([4, 8, 16])
    'n_modes': tune.grid_search([(16,16)]),
    'hidden_channels': tune.grid_search([16, 32]),
    'n_layers': tune.grid_search([4])
}

tuner = tune.Tuner(
    #tune.with_resources(training, resources={"gpu": 1}),
    training,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        scheduler=ASHAScheduler(metric="h1", mode="min"),
    )
)
print('Fitting now')
results = tuner.fit()
print('Fitting finished')

dfs = {result.path: result.metrics_dataframe for result in results}

fig, ax = plt.subplots()

for df in dfs.values():
    label = f"Modes{df['config/n_modes'][0]}, Channels{df['config/hidden_channels'][0]}, Layers{df['config/n_layers'][0]}"
    print(df)
    df['h1'].plot(ax=ax, label=label)

ax.set(xlabel='Epoch', ylabel='Error')
ax.set_xticks(range(1, len(df)+1))
ax.legend()
ax.set_title('H1 errors')

plt.savefig("combined_plots.png", bbox_inches='tight')
plt.close(fig)

bestModelDir = results.get_best_result('h1', mode='min').path
print(f'\nBest performing model is saved at {bestModelDir}')

ray.shutdown()