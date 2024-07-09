import numpy as np

import bandit.base

def fractal_sort(X):
  out = [X[0], X[-1]]
  bfs = [X[1:-1]]
  while bfs:
    chunk = bfs.pop(0)
    if len(chunk) <= 2:
      out += chunk
    else:
      mid = len(chunk) // 2
      out += [chunk[mid]]
      bfs += [chunk[:mid], chunk[mid+1:]]
  return out

class Fractal(bandit.base.Bandit):
  def calculate_state(self): 
    # std n ucb
    varying_columns = self.arms.apply(lambda x: x.nunique() > 1, axis=0)
    varying_columns = varying_columns[varying_columns].index
    assert varying_columns.shape[0] == 1, varying_columns
    arm = self.arms[varying_columns[0]]

    counts = self.rewards.groupby("arm_idx").size().rename("n")
    order = fractal_sort(arm)
    order = [
      x for x in order
      if x not in counts.index
    ]
    print(order)
    return order[0]