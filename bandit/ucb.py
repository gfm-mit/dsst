import time
import numpy as np
import pandas as pd

import util.config
import core.metrics

import bandit.base

def update_mean(mean, n, value):
  return (n * mean + value) / (n + 1)

class UCB(bandit.base.Bandit):
  def calculate_state(self): 
    # std n ucb
    counts = self.rewards.groupby("arm_idx").size().rename("n")
    state = counts.reindex(self.arms.index, fill_value=0)
    state["mu"] = self.rewards.auc.groupby("arm_idx").mean()
    state.mu = state.mu.fillna(-np.inf)
    state["sigma"] = self.rewards.auc.groupby("arm_idx").std(ddof=1)
    state.mu.fillna(state.sigma.nanmax(), inplace=True)
    
    with np.errstate(divide='ignore', invalid='ignore'):
      ucb = 4 * np.log(state.n.sum() - 1) / state.n
      ucb = 2 * state.sigma * np.sqrt(ucb)
    ucb[state.n == 0] = np.inf

    if self.args.task == "next_token":
      state.ucb = state.mu - ucb
      best_idx = state.ucb.argmin()
    else:
      state.ucb = state.mu + ucb
      best_idx = state.ucb.argmax()
    state.to_csv("results/bandit/state.csv")
    return best_idx