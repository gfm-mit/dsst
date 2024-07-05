import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.gaussian_process

import util.config
import core.metrics

def update_mean(mean, n, value):
  return (n * mean + value) / (n + 1)

def gp(args, experiment):
  K = "learning_rate"
  assert args.config
  setups = util.config.parse_config(args.config)
  assert not setups.duplicated().any()
  stats = pd.DataFrame(columns="X Y S".split())
  X_gpr = np.linspace(np.log(setups[K].min()), np.log(setups[K].max()), 100)[:, None]
  plt.ion()
  high, low = None, None
  kernel = sklearn.gaussian_process.kernels.RBF(length_scale_bounds=[3e-1,1e0]
    ) + sklearn.gaussian_process.kernels.WhiteKernel(noise_level_bounds=[1e-1,1e1])
  gpr = sklearn.gaussian_process.GaussianProcessRegressor(
    kernel=kernel, random_state=0, normalize_y=True, n_restarts_optimizer=10)
  fig, axs = plt.subplots(2, sharex=True)
  for iter in range(setups.shape[0]):
    params = setups.iloc[iter % setups.shape[0]]
    metric, epoch_loss_history = experiment.train(tqdm_prefix=None, **params.to_dict())
    steps = core.metrics.argbest(epoch_loss_history, args.task)
    stats.loc[stats.shape[0]] = params[K], metric, steps
    plt.sca(axs[1])
    plt.scatter(stats.X, stats.S, color="lightgray")
    plt.sca(axs[0])
    plt.scatter(stats.X, stats.Y, color="black")

    # Define the kernel: RBF for smooth changes, WhiteKernel for noise level
    gpr.fit(np.log(stats.X.values[:, None]), stats.Y)
    print(gpr.kernel_)
    Y_gpr, S_gpr = gpr.predict(X_gpr, return_std=True)
    if high is None:
      high, = plt.plot(np.exp(X_gpr), Y_gpr + 2 * S_gpr)
      low, = plt.plot(np.exp(X_gpr), Y_gpr - 2 * S_gpr)
      plt.xscale('log')
      plt.ylim([.4, .85])
    else:
      high.set_ydata(Y_gpr + 2 * S_gpr)
      low.set_ydata(Y_gpr - 2 * S_gpr)
    plt.pause(0.1)
  plt.ioff()
  plt.show()