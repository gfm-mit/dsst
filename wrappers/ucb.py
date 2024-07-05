import util.config


import numpy as np
import pandas as pd


import time


def ucb(args, experiment):
  assert args.config
  setups = util.config.parse_config(args.config)
  assert not setups.duplicated().any()
  stats = pd.DataFrame(index=setups.index)
  stats["n"] = 0
  stats["mu"] = 0.0
  stats["ucb"] = np.inf
  stats["mu2"] = 0.0
  stats["std"] = np.inf
  SD_GUESS = .01
  for iter in range(args.ucb):
    idx = stats.ucb.argmax()
    n, mu, ucb, mu2, std = stats.iloc[idx]
    params = setups.iloc[idx]
    seconds = time.time()
    metric, epoch_loss_history = experiment.train(tqdm_prefix=None, **params.to_dict())
    seconds = time.time() - seconds
    stats.loc[params.name, "mu"] = (n * mu + metric) / (n + 1)
    stats.loc[params.name, "mu2"] = (n * mu2 + metric**2) / (n + 1)
    stats.loc[params.name, "n"] = n + 1
    n, mu, ucb, mu2, std = stats.iloc[idx]
    if n > 1:
      stats.loc[params.name, "std"] = np.sqrt((mu2 - mu**2) / (n-1))

    if stats.n.min() == 0:
      stats.ucb = np.where(stats.n == 0, np.inf, -np.inf)
    else:
      stats.ucb = stats.mu + 2 * SD_GUESS * np.sqrt(4 * np.log(iter-1) / stats.n)
      stats.ucb = stats.ucb.fillna(np.inf)
    weight = stats.n - 1
    var = np.sum(stats.ucb.std()**2 * weight) / np.sum(weight)
    print(f"****** {iter+1}/{args.ucb}: {seconds:.2f}s, std: {np.sqrt(var)} *******")
    print(stats.sort_values(by="ucb"))
  print(f"****** FINAL *******")
  print(stats.mu.sort_values().tail(10))