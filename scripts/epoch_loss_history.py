import pathlib
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

down_metrics = set("rmse".split())
conf = pd.read_csv("results/bandit/conf.csv", index_col=0).iloc[0]
fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
for ax, label in zip(axs, "raw smooth max".split()):
  plt.sca(ax)
  plt.xlabel('epoch')
  plt.ylabel(label + " " + conf.metric)
for file in pathlib.Path("results/bandit/epoch").glob("*.npy"):
  epoch_loss_history = np.load(file)
  accum = np.minimum.accumulate if conf.metric in down_metrics else np.maximum.accumulate
  ylim = [0, 4]
  ylabel = conf.metric

  axs[0].plot(epoch_loss_history)
  axs[1].plot(scipy.ndimage.gaussian_filter1d(epoch_loss_history, sigma=2))
  axs[2].plot(accum(epoch_loss_history), label=file.stem)
plt.legend()
plt.show()