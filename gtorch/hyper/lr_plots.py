import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import gtorch.hyper.lr_finder

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

def show_lr(axs, losses, conds):
  losses = pd.DataFrame(losses).transpose()
  print(losses)
  conds = pd.DataFrame(conds).transpose()
  print(conds)
  plt.sca(axs[0])
  plt.xlabel("Learning Rate")
  plt.xscale('log')
  params = gtorch.hyper.lr_finder.get_lr_params()
  plt.xlim([params["min_lr"], params["max_lr"]])

  plt.ylabel("Loss")
  plt.yscale('log')
  plt.ylim(losses.min().min(), losses.iloc[0].max())

  plt.legend()
  plt.sca(axs[1])
  plt.xlabel("Learning Rate")
  plt.xscale('log')
  plt.xlim([params["min_lr"], params["max_lr"]])
  plt.ylabel('estimated condition')
  plt.yscale('log')
  plt.show()