import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


def set_ylim(data):
  q1, q3, p95, max = np.quantile(data, (0.25, 0.75, 0.95, 1))
  y_min, y_max = 2.5 * q1 - 1.5 * q3, 2 * max - p95
  plt.ylim([y_min, y_max])

def get_varying_params(X):
  constant = [
    k for k in X.columns
    if (X[k].iloc[0] == X[k]).all()
  ]
  XX = X.drop(columns=constant)
  return XX

def plot_history(args, epoch_loss_history, axs=None, label=None):
  accum = np.minimum.accumulate if args.stats == "train_loss" or args.task == "next_token" else np.maximum.accumulate
  ylabel = 'training loss' if args.stats == "train_loss" else 'validation auc' if args.task in "classify classify_patient classify_section".split() else 'validation rmse'
  if axs is None:
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
  plt.sca(axs[0])
  plt.plot(epoch_loss_history)
  plt.sca(axs[1])
  plt.plot(scipy.ndimage.gaussian_filter1d(epoch_loss_history, sigma=2))
  plt.sca(axs[2])
  plt.plot(accum(epoch_loss_history), label=label)
  plt.legend()

  for ax, label in zip(axs, "raw smooth max".split()):
    plt.sca(ax)
    plt.axhline(y=.715, color="lightgray", linestyle=":", zorder=-10)
    plt.axhline(y=.785, color="lightgray", linestyle=":", zorder=-10)
    plt.xlabel('epoch')
    plt.ylabel(label + ylabel)
    set_ylim(epoch_loss_history)
  return axs

def plot_best_values(X, y, task):
  assert isinstance(X, pd.DataFrame)
  assert isinstance(y, pd.Series)
  N = max(1, int(np.ceil(np.sqrt(X.shape[1]))))
  fig, axs = plt.subplots(N, N)
  if not isinstance(axs, np.ndarray):
    axs = [axs]
  else:
    axs = axs.flatten()
  metric_name = "RMSE" if task == "next_token" else "AUC-ROC-C"
  for e, k in enumerate(X.columns):
    if X[k].unique().shape[0] == 1: # yes, yes, inefficient
      continue
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

def print_and_plot_params(args, tuning_history, metric_history):
  tuning_history = get_varying_params(tuning_history)

  metric_history.index = tuning_history.index
  display_only = pd.concat([tuning_history, metric_history], axis=1)
  display_only.to_csv("results/params.csv")
  print(display_only)

  latex = display_only.groupby(tuning_history.columns.tolist()).aggregate(["mean", "std"])
  print("LaTeX & " + " & ".join(metric_history.columns))
  for idx, v in latex.iterrows():
    print(" & ".join([idx] + [
      f"${v[k]['mean']:.3f} \pm {v[k]['std']:.3f}$"
      for k in metric_history.columns
    ]))

  plot_metric = "rmse" if args.task == "next_token" else "auc"
  plot_metric = metric_history.loc[:, plot_metric].copy()
  plot_best_values(tuning_history, plot_metric, task=args.task)