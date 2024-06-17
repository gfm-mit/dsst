import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# todo, these two seem redundant
def plot_epoch_loss_history(args, epoch_loss_history, axs=None, label=None):
  ylabel = 'validation metric' if args.history == "val" else 'training loss'
  results = pd.Series(dict(history=epoch_loss_history, label=label)).to_frame().transpose()
  return plot_tuning_history(keys="label".split(), results=results, ylabel=ylabel, axs=axs)

def plot_tuning_history(keys, results, ylabel, axs=None):
  if axs is None:
    fig, axs = plt.subplots(1, 1)
  plt.sca(axs)
  for e, row in results.iterrows():
    label = str({
      k: "{:.2E}".format(row[k]) if isinstance(row[k], float) else row[k]
      for k in keys
    })
    plt.plot(row['history'], label=label)
  plt.legend()
  plt.axhline(y=.725, color="gray", zorder=-10)
  plt.xlabel('epoch')
  plt.ylabel(ylabel)
  y_min, y_max = .71, .725
  plt.ylim([y_min, y_max])
  return axs

def plot_tuning_results(keys, results, task):
  N = int(np.ceil(np.sqrt(len(keys))))
  fig, axs = plt.subplots(N, N)
  if not isinstance(axs, np.ndarray):
    axs = [axs]
  else:
    axs = axs.flatten()
  metric_name = "MSE" if task == "next_token" else "AUC-ROC-C"
  for e, k in enumerate(keys):
    plt.sca(axs[e])
    plt.scatter(results[k], results['metric'])
    plt.ylabel(metric_name)
    plt.xlabel(k)
    if results[k].dtype == object:
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