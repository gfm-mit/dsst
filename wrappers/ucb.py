import time
import numpy as np
import pandas as pd

import util.config
import core.metrics

def update_mean(mean, n, value):
  return (n * mean + value) / (n + 1)

def ucb(args, experiment, budget, resume=False):
  assert args.config
  setups = util.config.parse_config(args.config)
  assert not setups.duplicated().any()
  if resume:
    stats = pd.read_csv("results/ucb.csv", index_col=0)
    base = stats.n.sum()
    print(f"resuming after {base} steps")
    assert stats.index.equals(setups.index)
  else:
    stats = pd.DataFrame(index=setups.index)
    stats["mu2"] = 0.0
    stats["steps"] = 0.0
    stats["std"] = 0.0
    stats["n"] = 0
    stats["mu"] = 0.0
    stats["ucb"] = 0
    if args.task == "next_token":
      stats.ucb = -np.inf
    base = 0
  idx = 0
  for iter in range(budget):
    idx = fit_predict(args, budget, stats, base, iter)

    n, mu, mu2, steps = stats.iloc[idx]["n mu mu2 steps".split()]
    params = setups.iloc[idx]
    seconds = time.time()
    metric, epoch_loss_history = experiment.train(tqdm_prefix=None, **params.to_dict())
    steps_idx = core.metrics.argbest(epoch_loss_history, args.task)
    seconds = time.time() - seconds
    stats.loc[params.name, "mu"] = update_mean(mu, n, metric)
    stats.loc[params.name, "mu2"] = update_mean(mu2, n, metric**2)
    stats.loc[params.name, "n"] = n + 1
    stats.loc[params.name, "steps"] = update_mean(steps, n, steps_idx)

    n, mu, mu2, steps = stats.iloc[idx]["n mu mu2 steps".split()]
    if n > 1:
      stats.loc[params.name, "std"] = np.sqrt((mu2 - mu**2) / (n-1))
    print(f"********** {seconds:.2f}s **********")
  print(f"****** FINAL *******")
  print(stats.mu.sort_values().tail(10))

def fit_predict(args, budget, stats, base, iter):
    if stats.n.min() == 1 and stats.n.max() > 1:
      stats.loc[stats.n == 1, "std"] = stats["std"].max()

    with np.errstate(divide='ignore', invalid='ignore'):
      ucb_term = 4 * np.log(base + iter - 1) / stats.n
      optimism = 2 * stats["std"] * np.sqrt(ucb_term)
    optimism[stats.n == 0] = np.inf
    if args.task == "next_token":
      stats.ucb = stats.mu - optimism
      best_idx = stats.ucb.argmin()
    else:
      stats.ucb = stats.mu + optimism
      best_idx = stats.ucb.argmax()

    stats.to_csv("results/ucb.csv")
    print(stats.sort_values(by="ucb").drop(columns="mu2"))
    print(f"****** {base}+{iter+1}/{budget}: {best_idx=} *******")
    return best_idx