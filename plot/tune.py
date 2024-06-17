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

def main(train_loader, val_loader, builder=None, base_params=None, task="classify", disk="none", history="none"):
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
      task=task, disk=disk, tqdm_prefix=f"Tuning Case {i} {case_label}", history=history)
    results += [dict(**params, metric=metric, history=epoch_loss_history)]
  results = pd.DataFrame(results)
  if history == "none":
    results = results.drop(columns="history")
    plot_tuning_results(spaces, results, task)
  else:
    for e, row in results.iterrows():
      label = str({
        k: "{:.2E}".format(row[k]) if isinstance(row[k], float) else row[k]
        for k in spaces.keys()
      })
      plt.plot(row['history'], label=label)
    plt.legend()
    plt.axhline(y=.725, color="gray", zorder=-10)
    plt.xlabel('epoch')
    plt.ylabel(f'{history=}')
    #y_min = results.history.apply(lambda x: x[0]).min()
    #y_max = results.history.apply(max).max()
    y_min, y_max = .71, .725
    plt.ylim([y_min, y_max])
    plt.show()

def plot_tuning_results(spaces, results, task):
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
