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
    varying_columns = self.arms.apply(lambda x: x.nunique() > 1, axis=0)
    varying_columns = varying_columns[varying_columns].index
    assert varying_columns.shape[0] == 1, varying_columns # could do this in i_
    self.arm_1d = self.arms[varying_columns[0]]

  def calculate_state(self): 
    counts = self.rewards.groupby("arm_idx").size().rename("n")
    order = fractal_sort(self.arm_1d.copy()).drop(index=counts.index).index
    return order[0]