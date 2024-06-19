from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

import gtorch.datasets.linear_box
import gtorch.datasets.linear_patient
import gtorch.train.train
import gtorch.loss.optimizer
import plot.tune


def get_spaces(**kwargs):
  print("spaces.kwargs:")
  for k, v in kwargs.items():
    print(f"  {k}: {v}")
  spaces = pd.DataFrame(kwargs)
  for col in spaces.columns:
    spaces[col] = np.random.permutation(spaces[col].values)
  return spaces

def postprocess_tuning_ranges(tuning_ranges):
  for k in tuning_ranges.keys():
    if isinstance(tuning_ranges[k], dict):
      assert tuning_ranges[k].keys() == set("low high steps".split())
      tuning_ranges[k] = list(np.geomspace(tuning_ranges[k]["low"], tuning_ranges[k]["high"], tuning_ranges[k]["steps"]))
    else:
      assert isinstance(tuning_ranges[k], list)
  return tuning_ranges

def main(train_loader, val_loader, builder=None, base_params=None, task="classify", disk="none", history="none", tuning_ranges=None):
  torch.manual_seed(42)
  assert isinstance(builder, gtorch.models.base.Base)
  print(f"{tuning_ranges=}")
  if tuning_ranges is None:
    tuning_ranges = builder.get_tuning_ranges()
  else:
    tuning_ranges = postprocess_tuning_ranges(tuning_ranges)
  assert tuning_ranges
  spaces = get_spaces(**tuning_ranges)
  results = []
  for i in spaces.index:
    params = dict(**base_params)
    for k in spaces.columns:
      params[k] = spaces.loc[i, k]
    case_label = spaces.loc[i].to_dict()
    metric, epoch_loss_history, model = gtorch.train.train.setup_training_run(
      params, model_factory_fn=builder, train_loader=train_loader, val_loader=val_loader,
      task=task, disk=disk, tqdm_prefix=f"Tuning[{i+1}/{spaces.shape[0]}]={case_label}", history=history)
    results += [dict(**params, metric=metric, history=epoch_loss_history)]
  results = pd.DataFrame(results)
  if history == "none":
    results = results.drop(columns="history")
    plot.tune.plot_tuning_results(spaces.columns, results, task)
  else:
    plot.tune.plot_tuning_history(spaces.columns, results, ylabel=f"tune.args.{history=}")
    plt.show()