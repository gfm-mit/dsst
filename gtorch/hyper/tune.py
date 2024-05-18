import pandas as pd
import torch
from tqdm import tqdm

from etl.parse_semantics import *
from etl.parse_dynamics import *

from plot.palette import *
import gtorch.datasets.linear_agg
import gtorch.datasets.linear
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params

def get_spaces(**kwargs):
  spaces = pd.DataFrame(kwargs)
  for col in spaces.columns:
    spaces[col] = np.random.permutation(spaces[col].values)
  return spaces

def main(train_loader, val_loader, test_loader, axs=None, device='cpu', classes=2):
  torch.manual_seed(42)
  model, base_params = gtorch.models.linear.get_model(hidden_width=2, device=device, classes=classes)
  spaces = get_spaces(**base_params["tune"])
  results = []
  for i in tqdm(spaces.index):
    params = dict(**base_params)
    for k in spaces.columns:
      params[k] = spaces.loc[i, k]
    retval, model = gtorch.hyper.params.many_hyperparams(params, model_factory_fn=gtorch.models.linear.get_model,
                                                         train_loader=train_loader, val_loader=val_loader)
    results += [dict(**params, **retval)]
  results = pd.DataFrame(results)
  fig, axs = plt.subplots(1, 2)
  plt.sca(axs[0])
  plt.scatter(results.weight_decay, results.accuracy)
  plt.xscale('log')
  plt.xlabel('weight_decay')
  plt.sca(axs[1])
  plt.scatter(results.learning_rate, results.accuracy)
  plt.xscale('log')
  plt.xlabel('learning_rate')
  plt.show()
  print(results)
  return axs, None
