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
    K="learning_rate", scale="log", budget=None, resume=False):
  assert args.config
  setups = util.config.parse_config(args.config)
  setups = setups.sample(frac=1)
  assert not setups.duplicated().any()
  if resume:
    stats = pd.read_csv("results/gp.csv", index_col=0)
  else:
    stats = pd.DataFrame(columns="X Y S".split())
  #gpr = wrappers.gpr.GPR(K, scale, budget, setups[K].min(), setups[K].max())
  #fig, axs = plt.subplots(2, sharex=True)
  #plt.ion()
  for iter in range(budget):
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
    print(f"kernel[{iter+1}/{budget}]={delay:.2f}s")
    #gpr.fit(stats)
    #gpr.update_plot(axs)
    #gpr.scatter(stats, axs) # this might actually leak.  dammit
    #plt.pause(0.1)
  #plt.ioff()
  print(stats.groupby("X").mean().sort_index())
  #plt.show()