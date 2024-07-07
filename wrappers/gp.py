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
    K=None, scale="log", budget=None, resume=False):
  assert args.config
  setups = util.config.parse_config(args.config)
  idx = [
    setups.shape[0] - 1, 0,
    *np.random.default_rng().permuted(np.arange(1, setups.shape[0] - 1))
  ]
  setups = setups.iloc[idx]
  assert not setups.duplicated().any()
  nonconstant_columns = [col for col in setups.columns if setups[col].nunique() > 1]
  assert len(nonconstant_columns) == 1, nonconstant_columns
  K = nonconstant_columns[0]

  if resume:
    stats = pd.read_csv("results/gp.csv", index_col=0)
  else:
    stats = pd.DataFrame(columns="X Y S".split())
  pd.Series(dict(K=K, scale=scale, budget=budget, min=setups[K].min(), max=setups[K].max(), task=args.task)).to_frame().transpose().to_csv("results/gp_args.csv")
  gpr = wrappers.gpr.GPR(K, scale, budget, setups[K].min(), setups[K].max())
  for iter in range(budget):
    gpr.fit(stats)
    gpr.predict()
    print(gpr.pick(setups[K].values))
    params = setups.iloc[iter % setups.shape[0]]
    try:
      delay = time.time()
      metric, epoch_loss_history = experiment.train(tqdm_prefix=None, **params.to_dict())
      delay = time.time() - delay
    except KeyboardInterrupt:
      print("KeyboardInterrupt")
      break
    steps = core.metrics.argbest(epoch_loss_history, args.task)
    stats.loc[stats.shape[0]] = params[K], metric, steps
    stats.to_csv("results/gp.csv")
    print(f"kernel[{iter+1}/{budget}]({params[K]})={delay:.2f}s")
  print(stats.groupby("X").mean().sort_index())