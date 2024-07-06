import time
import numpy as np
import pandas as pd

import util.config
import core.metrics

def update_mean(mean, n, value):
  return (n * mean + value) / (n + 1)

def ucb(args, experiment, budget):
  assert args.config
  setups = util.config.parse_config(args.config)
  assert not setups.duplicated().any()
  stats = pd.DataFrame(index=setups.index)
  stats["mu2"] = 0.0
  stats["steps"] = 0.0
  stats["std"] = np.inf
  stats["n"] = 0
  stats["mu"] = 0.0
  stats["ucb"] = np.inf
  SD_GUESS = .01
  for iter in range(budget):
    idx = stats.ucb.argmax()
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

    if stats.n.min() == 0:
      stats.ucb = np.where(stats.n == 0, np.inf, -np.inf)
    elif args.task == "next_token":
      stats.ucb = -stats.mu + 2 * SD_GUESS * np.sqrt(4 * np.log(iter-1) / stats.n)
      #stats.ucb = stats.ucb.fillna(np.inf)
    else:
      stats.ucb = stats.mu + 2 * SD_GUESS * np.sqrt(4 * np.log(iter-1) / stats.n)
      #stats.ucb = stats.ucb.fillna(np.inf)
    weight = stats.n - 1
    var = np.where(stats.n >= 2, stats['std']**2, 0)
    var = np.sum(var * weight) / np.sum(weight)
    print(f"****** {iter+1}/{budget}: {seconds:.2f}s, std: {np.sqrt(var):.5f} *******")
    print(stats.sort_values(by="ucb").drop(columns="mu2"))
  print(f"****** FINAL *******")
  print(stats.mu.sort_values().tail(10))