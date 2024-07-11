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

def gpr(rewards, arms, config):
  assert arms.shape[1] == 1, arms.columns
  K = arms.columns[0]
  stats = rewards.join(arms, on="arm_idx")[[K] + [config["metric"]] + ["best_epoch"]]
  stats.columns = "X Y S".split()
  config["scale"] = "log"
  config["min"] = arms.min()
  config["max"] = arms.max()
  gpr = plot.gpr.GPR(**config)
  targets = gpr.get_default_targets(**config)
  gpr.cv_fit(stats)["kernel__length_scale"]
  targets, best_idx = gpr.fit_predict(stats, targets=targets)
  axs = gpr.make_plot(targets, best_idx, axs=None)
  gpr.scatter(stats, axs=axs)
  plt.draw()

def ucb(rewards, arms, config):
  df = rewards.join(arms, on="arm_idx")
  print(df.groupby(arms.columns.tolist()).agg("mean std".split()).sort_values(by=(config["metric"],"mean")))

  model = sklearn.linear_model.RidgeCV()
  pairwise = pd.concat([
    pd.get_dummies(df[g].astype('category'), prefix=g, drop_first=True)
    for g in arms.columns
  ],axis=1)
  i, j = np.triu_indices(pairwise.shape[1], k=0)
  unfolded = np.stack([
    pairwise.values[:, ii] * pairwise.values[:, jj]
    for ii, jj in zip(i, j)
  ]).T
  model.fit(unfolded, df[config["metric"]])
  folded = np.eye(pairwise.shape[1]) + np.nan
  for i, j, c, v in zip(i, j, model.coef_, unfolded.var(axis=0)):
    if v > 0:
      folded[i, j] = c
  folded = pd.DataFrame(folded, index=pairwise.columns, columns=pairwise.columns)
  sns.heatmap(folded, center=0, cmap='coolwarm')
  plt.title(
    df[arms.columns].astype('category').apply(lambda x: x.cat.categories[0]).rename('base')
  )
  plt.tight_layout()
  plt.draw()

if __name__ == "__main__":
  while True:
    rewards = pd.read_csv('results/bandit/rewards.csv', index_col=0)
    arms = pd.read_csv('results/bandit/arms.csv', index_col=0)
    arms = bandit.base.get_varying_columns(arms)
    config = pd.read_csv('results/bandit/conf.csv', index_col=0).iloc[0].to_dict()
    if rewards.shape[0] == 1:
      plt.title("only one arm")
      plt.draw()
    elif arms.shape[1] == 1 and arms.iloc[:, 0].dtype == float:
      gpr(rewards, arms, config)
    else:
      ucb(rewards, arms, config)
    try:
      print("start wait")
      plt.waitforbuttonpress(0)
      print("finish wait")
    except KeyboardInterrupt:
      break
    finally:
      print("close")
      plt.close()