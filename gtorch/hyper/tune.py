import pandas as pd
import torch
from tqdm import tqdm

from etl.parse_semantics import *
from etl.parse_dynamics import *

from plot.palette import *
import gtorch.datasets.linear_patient
import gtorch.datasets.linear_box
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params

def get_spaces(**kwargs):
  print("spaces.kwargs", kwargs)
  spaces = pd.DataFrame(kwargs)
  for col in spaces.columns:
    spaces[col] = np.random.permutation(spaces[col].values)
  return spaces

def main(train_loader, val_loader, test_loader, axs=None, device='cpu', classes=2, builder=None):
  torch.manual_seed(42)
  assert isinstance(builder, gtorch.models.base.Base)
  base_params = builder.get_parameters()
  assert builder.get_tuning_ranges(), "no parameters to tune"
  spaces = get_spaces(**builder.get_tuning_ranges())
  results = []
  for i in tqdm(spaces.index):
    params = dict(**base_params)
    for k in spaces.columns:
      params[k] = spaces.loc[i, k]
    print("tune:", spaces.loc[i].to_dict())
    retval, model = gtorch.hyper.params.setup_training_run(params, model_factory_fn=builder,
                                                           train_loader=train_loader, val_loader=val_loader)
    results += [dict(**params, **retval)]
  results = pd.DataFrame(results)
  N = int(np.ceil(np.sqrt(spaces.shape[1])))
  fig, axs = plt.subplots(N, N)
  if not isinstance(axs, np.ndarray):
    axs = [axs]
  else:
    axs = axs.flatten()
  for e, k in enumerate(spaces.columns):
    plt.sca(axs[e])
    plt.scatter(results[k], results.roc)
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
