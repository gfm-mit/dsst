import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import gtorch.datasets.linear_box
import gtorch.datasets.linear_patient
import gtorch.train.train
import gtorch.loss.optimizer


def get_spaces(**kwargs):
  print("spaces.kwargs:")
  for k, v in kwargs.items():
    print(f"  {k}: {v}")
  spaces = pd.DataFrame(kwargs)
  for col in spaces.columns:
    spaces[col] = np.random.permutation(spaces[col].values)
  return spaces

def main(train_loader, val_loader, builder=None, base_params=None, task="classify", disk="none"):
  torch.manual_seed(42)
  assert isinstance(builder, gtorch.models.base.Base)
  assert builder.get_tuning_ranges(), "no parameters to tune"
  spaces = get_spaces(**builder.get_tuning_ranges())
  results = []
  for i in spaces.index:
    params = dict(**base_params)
    for k in spaces.columns:
      params[k] = spaces.loc[i, k]
    case_label = spaces.loc[i].to_dict()
    metric, epoch_loss_history, model = gtorch.train.train.setup_training_run(
      params, model_factory_fn=builder, train_loader=train_loader, val_loader=val_loader,
      task=task, disk=disk, tqdm_prefix=f"Tuning Case {i} {case_label}")
    results += [dict(**params, metric=metric)]
  results = pd.DataFrame(results)
  print(results)
  N = int(np.ceil(np.sqrt(spaces.shape[1])))
  fig, axs = plt.subplots(N, N)
  if not isinstance(axs, np.ndarray):
    axs = [axs]
  else:
    axs = axs.flatten()
  metric_name = "MSE" if task == "next_token" else "AUC-ROC-C"
  for e, k in enumerate(spaces.columns):
    plt.sca(axs[e])
    plt.scatter(results[k], results['metric'])
    plt.ylabel(metric_name)
    plt.xlabel(k)
    if results[k].dtype == str:
      plt.xticks(rotation=45)
    elif 0 < results[k].min() < results[k].max() < 1:
      plt.xscale('logit')
    elif 0 < results[k].min() < 1:
      plt.xscale('log')
    else:
      pass
  plt.suptitle("Aggregated at Box Level, not Patient Level")
  plt.show()
  print(results)
  return axs, None
