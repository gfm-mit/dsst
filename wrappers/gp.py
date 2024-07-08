import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import util.config
import core.metrics
import wrappers.experiment
import wrappers.gpr

def update_mean(mean, n, value):
  return (n * mean + value) / (n + 1)

def gp(
    args, experiment: wrappers.experiment.Experiment,
    K=None, scale="log", budget=None, resume=False, sigma=0.1):
  assert args.config
  setups = util.config.parse_config(args.config)
  assert not setups.duplicated().any()
  nonconstant_columns = [col for col in setups.columns if setups[col].nunique() > 1]
  assert len(nonconstant_columns) == 1, nonconstant_columns
  K = nonconstant_columns[0]

  if resume:
    stats = pd.read_csv("results/gp.csv", index_col=0)
  else:
    stats = pd.DataFrame(columns="X Y S".split())
  pd.Series(dict(K=K, scale=scale, budget=budget, task=args.task, min=setups[K].min(), max=setups[K].max(), sigma=sigma)).to_frame().transpose().to_csv("results/gp_args.csv")
  gpr = wrappers.gpr.GPR(K, scale, budget, sigma=sigma)
  fractal_order = fractal_sort(np.arange(setups.shape[0]).tolist())
  if len(fractal_order) > 3: # should help fit the white kernel
    fractal_order = fractal_order[:2] + fractal_order[2:3] * 3 + fractal_order[3:]
  for iter in range(budget):
    targets = pd.DataFrame(setups[K].values.copy(), columns=["X"])
    if sigma <= 0:
      best_idx = fractal_order[iter % len(fractal_order)]
      print(f"fractal {best_idx=} {targets.X.iloc[best_idx]=}")
    else:
      targets, best_idx = gpr.fit_predict(stats, targets=targets)
      print(f"gp {best_idx=} {targets.X.iloc[best_idx]=}")
    params = setups.iloc[best_idx]
    try:
      delay = time.time()
      metric, epoch_loss_history = experiment.train(tqdm_prefix=None, **params.to_dict())
      delay = time.time() - delay
    except KeyboardInterrupt:
      print("KeyboardInterrupt")
      break
    steps = core.metrics.argbest(epoch_loss_history, args.task)
    new_idx = stats.index.max() + 1 if stats.shape[0] > 0 else 0
    stats.loc[new_idx] = params[K], metric, steps
    stats.to_csv("results/gp.csv")
    print(f"kernel[{iter+1}/{budget}]({params[K]})={delay:.2f}s")
  print(stats.groupby("X").mean().sort_index())

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