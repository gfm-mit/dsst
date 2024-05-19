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
  assert "tune" in base_params and len(base_params["tune"]) > 0, "no parameters to tune"
  spaces = get_spaces(**base_params["tune"])
  results = []
  for i in tqdm(spaces.index):
    params = dict(**base_params)
    for k in spaces.columns:
      params[k] = spaces.loc[i, k]
    print("tune:", spaces.loc[i].to_dict())
    retval, model = gtorch.hyper.params.many_hyperparams(params, model_factory_fn=gtorch.models.linear.get_model,
                                                         train_loader=train_loader, val_loader=val_loader)
    results += [dict(**params, **retval)]
  results = pd.DataFrame(results)
  fig, axs = plt.subplots(1, len(base_params["tune"]))
  if not isinstance(axs, np.ndarray):
    axs = [axs]
  for e, k in enumerate(base_params["tune"].keys()):
    plt.sca(axs[e])
    plt.scatter(results[k], results.accuracy)
    plt.xlabel(k)
    if results[k].max() < 1:
      plt.xscale('logit')
    elif 0 < results[k].min() < 1:
      plt.xscale('log')
    else:
      pass
  plt.show()
  print(results)
  return axs, None
