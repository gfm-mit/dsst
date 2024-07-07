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
  config = pd.read_csv('results/gp_args.csv', index_col=0).iloc[0]
  gpr = wrappers.gpr.GPR(**config)
  gpr.fit(stats)
  fig, axs = plt.subplots(2, sharex=True)
  gpr.update_plot(axs)
  gpr.scatter(stats, axs)
  plt.draw()
  try:
    plt.waitforbuttonpress(0)
  except KeyboardInterrupt:
    break
  finally:
    plt.close()