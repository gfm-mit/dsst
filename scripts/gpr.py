# flake8: noqa
import argparse
import pprint
import re
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.gaussian_process
import tomli

import plot.gpr
import bandit.base

axs = None
while True:
  rewards = pd.read_csv('results/bandit/rewards.csv', index_col=0)
  arms = pd.read_csv('results/bandit/arms.csv', index_col=0)
  arms = bandit.base.get_varying_columns(arms)
  assert arms.shape[1] == 1, arms.columns
  K = arms.columns[0]
  stats = rewards.join(arms, on="arm_idx")[[K] + "auc best_epoch".split()]
  stats.columns = "X Y S".split()
  config = pd.read_csv('results/bandit/conf.csv', index_col=0).iloc[0].to_dict()
  config["scale"] = "log"
  config["sigma"] = .2
  config["K"] = K
  config["task"] = "classify"
  config["min"] = arms.min()
  config["max"] = arms.max()
  gpr = plot.gpr.GPR(**config)
  targets = gpr.get_default_targets(**config)
  targets, best_idx = gpr.fit_predict(stats, targets=targets)
  axs = None # fuck it
  axs = gpr.update_plot(targets, best_idx, axs=axs)
  gpr.scatter(stats, axs=axs)
  plt.draw()
  try:
    plt.waitforbuttonpress(0)
  except KeyboardInterrupt:
    break
  finally:
    plt.close()