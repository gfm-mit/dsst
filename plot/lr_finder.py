import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

def plot_lr(lrs, losses, smooth=3):
  # pretty sure these two operators commute
  losses = scipy.ndimage.gaussian_filter1d(losses, smooth / 2, mode='nearest')
  deltas = np.diff(losses)
  thresh = np.nanmedian(deltas[deltas > 0]) - np.nanmedian(deltas[deltas < 0])
  pd.Series(deltas, index=lrs[1:]).plot()
  plt.yscale('symlog', linthresh=thresh)
  plt.xscale('log')
  plt.axhline(0, color='lightgray', zorder=-10, linestyle='--')
  plt.show()