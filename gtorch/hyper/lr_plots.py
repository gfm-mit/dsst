import matplotlib.pyplot as plt
import pandas as pd
import scipy

def plot_lr(lrs, losses, smooth=None, label=None, ax=None):
  # pretty sure these two operators commute
  if smooth is not None:
    losses = scipy.ndimage.gaussian_filter1d(losses, smooth / 2, mode='nearest')
  losses = pd.Series(losses, index=lrs, name=label)
  ax = losses.plot(ax=ax, label=label)
  plt.xscale('log')
  plt.yscale('log')
  return losses, ax

def show_lr(ax, losses):
  print(pd.DataFrame(losses).transpose())
  plt.xlabel("Learning Rate")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()