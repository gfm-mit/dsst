import time
import numpy as np
import pandas as pd

import util.config

class Bandit:
  def __init__(self, conf: dict, arms: pd.DataFrame):
    super().__init__()
    self.conf = conf
    pd.Series(self.conf).to_frame().T.to_csv("results/bandit/conf.csv")
    self.arms = arms.reset_index(drop=True)
    if conf["resume"]:
      old_arms = pd.read_csv("results/bandit/arms.csv", index_col=0)
      assert old_arms.equals(self.arms)
      self.rewards = pd.read_csv("results/bandit/rewards.csv", index_col=0)
    else:
      self.rewards = None
    self.arms.to_csv("results/bandit/arms.csv")
    self.metric = conf["metric"]
  
  def suggest_arm(self):
    if self.rewards is None: # TODO if there's any algorithm that doesn't start with option 0
      self.idx = self.arms.index[0]
      return self.arms.loc[self.idx]
    if self.rewards.shape[0] >= self.conf["budget"]:
      return None
    self.idx = self.calculate_state()
    return self.arms.loc[self.idx]
  
  def update_rewards(self, metric):
    if self.rewards is None:
      self.rewards = pd.Series(metric).rename(0).to_frame().transpose()
      self.rewards["arm_idx"] = self.idx
      print(f"Rewards:\n{self.rewards}")
    else:
      self.rewards.loc[self.rewards.index.max() + 1] = dict(**metric, arm_idx=self.idx)
    self.rewards.to_csv("results/bandit/rewards.csv")
  
  def calculate_state(self): 
    counts = self.rewards.groupby("arm_idx").size().rename("n")
    state = counts.reindex(self.arms.index, fill_value=0)
    state.to_csv("results/bandit/state.csv")
    return state.n.idxmin()

  def get_varying_columns(self):
    return get_varying_columns(self.arms)

  def get_label(self):
    label = {}
    for k in self.arms.columns:
      if self.arms[k].nunique() != 1:
        label[k] = self.arms.loc[self.idx, k]
    return util.config.pprint_dict(label)

def get_varying_columns(arms):
  columns = [
    k for k in arms.columns
    if arms[k].nunique() > 1
  ]
  return arms[columns]