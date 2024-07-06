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

import wrappers.gpr

while True:
  stats = pd.read_csv('results/gp.csv')
  with open('config/sop.toml', 'rb') as f:
    config = tomli.load(f)["meta"]["bandit"]
  gpr = wrappers.gpr.GPR(config['K'], config['scale'], config['budget'], stats.X.min(), stats.X.max())
  gpr.fit(stats)
  fig, axs = plt.subplots(2, sharex=True)
  gpr.update_plot(axs)
  gpr.scatter(stats, axs)
  plt.draw()
  plt.waitforbuttonpress(0)
  plt.close()