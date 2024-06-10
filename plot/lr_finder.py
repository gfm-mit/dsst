import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_lr(lrs, losses):
  deltas = np.diff(losses)
  min_delta = np.nanmin(np.abs(deltas))
  pd.Series(deltas, index=lrs[1:]).plot()
  plt.yscale('symlog', linthresh=min_delta)
  plt.xscale('log')
  plt.show()