import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_ylim(data):
  low, med, high = np.quantile(data, (0.25, 0.95, 1))
  y_min, y_max = 2.5 * low - 1.5 * high, 2 * high - med
  y_min = np.maximum(y_min, 0.5)
  plt.ylim([y_min, y_max])

def plot_history(args, epoch_loss_history, axs=None, label=None):
  ylabel = 'training loss' if args.history == "loss" else 'validation metric'
  if axs is None:
    fig, axs = plt.subplots(1, 1)
  plt.sca(axs)
  plt.plot(epoch_loss_history, label=label)
  plt.legend()
  plt.axhline(y=.715, color="lightgray", linestyle=":", zorder=-10)
  plt.axhline(y=.785, color="lightgray", linestyle=":", zorder=-10)
  plt.xlabel('epoch')
  plt.ylabel(ylabel)
  set_ylim(epoch_loss_history)
  return axs

def plot_best_values(X, y, task):
  assert isinstance(X, pd.DataFrame)
  assert isinstance(y, pd.Series)
  print(pd.concat([X, y], axis=1))
  N = int(np.ceil(np.sqrt(X.shape[1])))
  fig, axs = plt.subplots(N, N)
  if not isinstance(axs, np.ndarray):
    axs = [axs]
  else:
    axs = axs.flatten()
  metric_name = "RMSE" if task == "next_token" else "AUC-ROC-C"
  for e, k in enumerate(X.columns):
    plt.sca(axs[e])
    plt.scatter(X[k], y)
    plt.ylabel(metric_name)
    plt.xlabel(k)
    if X[k].dtype == object:
      plt.xticks(rotation=45)
    elif 0 < X[k].min() < X[k].max() < 1:
      plt.xscale('logit')
    elif 0 < X[k].min() < 1:
      plt.xscale('log')
    else:
      pass
  plt.tight_layout()
  plt.show()