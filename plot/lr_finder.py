import matplotlib.pyplot as plt
import pandas as pd
import scipy

def plot_lr(lrs, losses, smooth=3, ax=None):
  # pretty sure these two operators commute
  losses = scipy.ndimage.gaussian_filter1d(losses, smooth / 2, mode='nearest')
  losses = pd.Series(losses, index=lrs)
  losses.plot(ax=ax)
  plt.xscale('log')
  plt.yscale('log')
  return losses