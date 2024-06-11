import matplotlib.pyplot as plt
import pandas as pd
import scipy

def plot_lr(lrs, losses, conds=None, smooth=None, label=None, axs=None):
  # pretty sure these two operators commute
  if smooth is not None:
    losses = scipy.ndimage.gaussian_filter1d(losses, smooth / 2, mode='nearest')
    conds = scipy.ndimage.gaussian_filter1d(conds, smooth / 2, mode='nearest')
  losses = pd.Series(losses, index=lrs, name=label)
  losses.plot(ax=axs[0], label=label)
  conds = pd.Series(conds, index=lrs[1:], name=label + " K")
  conds.plot(ax=axs[1], label=conds.name)
  return losses, conds

def show_lr(axs, losses, conds):
  losses = pd.DataFrame(losses).transpose()
  #print(losses)
  conds = pd.DataFrame(conds).transpose()
  #print(conds)
  plt.sca(axs[0])
  plt.xlabel("Learning Rate")
  plt.xscale('log')

  plt.ylabel("Loss")
  plt.yscale('log')
  plt.ylim(losses.min().min(), losses.iloc[0].max())

  plt.legend()
  plt.sca(axs[1])
  plt.xlabel("Learning Rate")
  plt.xscale('log')
  plt.ylabel('estimated condition')
  plt.yscale('log')
  plt.show()