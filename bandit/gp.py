import numpy as np
import pandas as pd

import bandit.base
import bandit.fractal_1d
import plot.gpr

class GP(bandit.base.Bandit):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    varying_columns = self.get_varying_columns()
    assert varying_columns.shape[1] == 1, varying_columns.columns
    self.arm_1d = varying_columns.iloc[:, 0]

    self.gpr = plot.gpr.GPR(
      scale=self.conf["scale"],
      budget='unused',
      sigma=self.conf["sigma"],
      task=self.conf["task"])

  def calculate_state(self): 
    X = self.arm_1d.rename("X").to_frame()
    Y = self.rewards[self.metric].rename("Y").to_frame()
    stats = X.join(Y, how='inner')
    targets = X.copy()
    targets["Y"] = np.nan
    targets, best_idx = self.gpr.fit_predict(stats, targets=targets)
    return targets.iloc[best_idx].name