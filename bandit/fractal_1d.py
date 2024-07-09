import numpy as np
import pandas as pd

import bandit.base

def fractal_sort(X):
  X = X.sort_values()
  out = [X.iloc[:1], X.iloc[-1:]]
  bfs = [X.iloc[1:-1]]
  while bfs:
    chunk = bfs.pop(0)
    if chunk.shape[0] <= 2:
      out += [chunk]
    else:
      mid = chunk.shape[0] // 2
      out += [chunk.iloc[mid:mid+1]]
      bfs += [chunk.iloc[:mid], chunk.iloc[mid+1:]]
  out = pd.concat(out)
  return out

class Fractal(bandit.base.Bandit):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    varying_columns = self.get_varying_columns()
    assert varying_columns.shape[1] == 1, varying_columns.columns
    self.arm_1d = varying_columns.iloc[:, 0]

  def calculate_state(self): 
    counts = self.rewards.groupby("arm_idx").size().rename("n")
    order = fractal_sort(self.arm_1d.copy()).drop(index=counts.index).index
    return order[0]