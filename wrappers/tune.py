from matplotlib import pyplot as plt
import pandas as pd
import torch

import models
import core.metrics
import core.train
import core.optimizer
import plot.tune
from util.config import get_spaces
from util.config import postprocess_tuning_ranges


def pprint_dict(d):
  str_dict = {}
  for k, v in d.items():
      if isinstance(v, str):
        str_dict[k] = v
      elif isinstance(v, int):
        str_dict[k] = str(v)
      elif 0 < v < .1:
        str_dict[k] = f"{v:.1e}"
      elif 0.9 < v < 1:
        str_dict[k] = f"1 - {1- v:.0e}"
      elif 0.1 <= v <= .9 or v == 0:
        str_dict[k] = f"{v:.2f}"
      else:
        str_dict[k] = f"{v:.2e}"
  return str(str_dict)

def main(train_loader, val_loader, builder=None, base_params=None, task="classify", disk="none", history="none", tuning_ranges=None):
  torch.manual_seed(42)
  assert isinstance(builder, models.base.Base)
  assert tuning_ranges
  overlay, tuning_ranges = postprocess_tuning_ranges(tuning_ranges)
  print(f"{tuning_ranges=}")
  assert tuning_ranges
  spaces = get_spaces(**tuning_ranges)
  results = []
  for i in spaces.index:
    params = dict(**base_params) | spaces.loc[i].to_dict() | overlay
    case_label = pprint_dict(spaces.loc[i].to_dict())
    metric, epoch_loss_history, model = core.train.setup_training_run(
      params, model_factory_fn=builder, train_loader=train_loader, val_loader=val_loader,
      task=task, disk=disk, tqdm_prefix=f"Tuning[{i+1}/{spaces.shape[0]}]={case_label}",
      use_loss_history=False)
    metric = core.metrics.early_stop(epoch_loss_history, task=task)
    results += [dict(**params, metric=metric, history=epoch_loss_history)]
  results = pd.DataFrame(results)
  if spaces.columns.size == 1:
    results = results.sort_values(by=spaces.columns[0])
  if history == "none":
    plot.tune.plot_tuning_results(spaces.columns, results, task)
  else:
    plot.tune.plot_tuning_history(spaces.columns, results, ylabel=f"tune.args.{history=}")
    plt.show()