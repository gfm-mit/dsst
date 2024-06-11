import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


def plot_lr(lrs, losses, conds=None, smooth=None, label=None, axs=None):
  losses = pd.Series(losses, index=lrs, name=label)
  conds = pd.Series(conds, index=lrs, name="K " + label)
  # pretty sure these two operators commute
  if smooth is not None:
    losses = pd.Series(
       scipy.ndimage.gaussian_filter1d(losses.values, smooth / 2, mode='nearest'),
       index=losses.index, name=losses.name)
    conds = conds.replace([np.inf, -np.inf], np.nan).dropna()
    conds = pd.Series(
       scipy.ndimage.gaussian_filter1d(conds.values, smooth / 2, mode='nearest'),
       index=conds.index, name=conds.name)
  losses.plot(ax=axs[0], label=label)
  axs[1].scatter(conds.index, conds, label=conds.name)
  return losses, conds

def get_axes(params):
  fig, axs = plt.subplots(2, 1)
  plt.sca(axs[0])
  plt.xlabel("Learning Rate")
  plt.xscale('log')
  plt.xlim([params["min_lr"], params["max_lr"]])

  plt.ylabel("Loss")
  plt.yscale('log')

  plt.sca(axs[1])
  plt.xlabel("Learning Rate")
  plt.xscale('log')
  plt.xlim([params["min_lr"], params["max_lr"]])
  plt.ylabel('estimated condition')
  plt.yscale('log')
  return axs

def show_axes(axs, losses, conds):
  losses = pd.DataFrame(losses).transpose()
  conds = pd.DataFrame(conds).transpose()
  print(pd.concat([losses, conds], axis=1))
  maxes = []
  for c in losses:
    monotonic_tail_length = (losses[c].diff() < 0).iloc[::-1].argmax()
    if monotonic_tail_length:
      interior_max = np.nanmax(losses[c].iloc[:-monotonic_tail_length])
    else:
      interior_max = np.nanmax(losses[c])
    print(monotonic_tail_length)
    maxes += [interior_max]
  low = losses.min().min()
  high = np.nanmax(maxes)
  print(low, high)

  plt.sca(axs[0])
  plt.ylim([1.1 * low - .1 * high, 1.1 * high - .1 * low])
  plt.legend()
  plt.tight_layout()
  plt.show()