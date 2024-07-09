import time
import numpy as np
import pandas as pd

import util.config
import core.metrics

class Bandit:
  def __init__(self, args: dict, conf: dict, arms: pd.DataFrame):
    super().__init__()
    self.args = args
    pd.Series(self.args).to_csv("results/bandit/args.csv")
    self.conf = conf
    pd.Series(self.conf).to_csv("results/bandit/conf.csv")
    self.arms = arms.reset_index(drop=True)
    if conf["resume"]:
      old_arms = pd.read_csv("results/bandit/arms.csv", index_col=0)
      assert old_arms.equals(self.arms)
      self.rewards = pd.read_csv("results/bandit/rewards.csv", index_col=0)
    self.arms.to_csv("results/bandit/arms.csv")
  
  def suggest_arm(self):
    if self.rewards.shape[0] >= self.conf["budget"]:
      return None
    self.state = self.calculate_state()
    self.idx = self.state.n.idxmin()
    return self.arms.loc[self.idx]
  
  def update_rewards(self, metric):
    if self.rewards is None:
      self.rewards = metric.rename(0).to_frame().transpose()
      self.rewards["arm_idx"] = self.idx
    else:
      self.rewards.loc[self.rewards.index.max() + 1] = dict(**metric, arm_idx=self.idx)
    self.rewards.to_csv("results/bandit/rewards.csv")
  
  def calculate_state(self): 
    counts = self.rewards.groupby("arm_idx").size().rename("n")
    return counts.reindex(self.arms.index, fill_value=0)

  def get_label(self):
    label = {}
    for k in self.arms.columns:
      if self.arms[k].nunique() != 1:
        label[k] = self.arms.loc[self.idx, k]
    return util.config.pprint_dict(label)