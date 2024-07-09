import time
import numpy as np
import pandas as pd

import util.config
import core.metrics

class Bandit:
  def __init__(self, args: dict, conf: dict, options: pd.DataFrame):
    super().__init__()
    self.args = args
    pd.Series(self.args).to_csv("results/bandit/args.csv")
    self.conf = conf
    pd.Series(self.conf).to_csv("results/bandit/conf.csv")
    self.options = options.reset_index(drop=True)
    self.options.to_csv("results/bandit/options.csv")
    self.results = None
  
  def suggest_option(self):
    if self.results.shape[0] > self.conf["budget"]:
      return None
    self.update_state()
    self.idx = self.state.n.idxmin()
    return self.options.loc[self.idx]
  
  def update_results(self, metric):
    if self.results is None:
      self.results = metric.rename(0).to_frame().transpose()
      self.results["option_idx"] = self.idx
    else:
      self.results.loc[self.results.index.max() + 1] = dict(**metric, option_idx=self.idx)
    self.results.to_csv("results/bandit/results.csv")
  
  def update_state(self): 
    counts = self.results.groupby("option_idx").size().rename("n")
    self.state = counts.reindex(self.options.index, fill_value=0)