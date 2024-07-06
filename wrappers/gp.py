import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.gaussian_process

import util.config
import core.metrics
import wrappers.experiment

def update_mean(mean, n, value):
  return (n * mean + value) / (n + 1)

def gp(
    args, experiment: wrappers.experiment.Experiment,
    K="learning_rate", scale="log", budget=None):
  assert args.config
  setups = util.config.parse_config(args.config)
  assert not setups.duplicated().any()
  stats = pd.DataFrame(columns="X Y S E V".split())
  low, high = setups[K].min(), setups[K].max()
  if scale == "log":
    X_gpr = np.geomspace(low, high, 100)[:, None]
    to_gp_space = np.log
  elif scale == "log1p":
    X_gpr = np.geomspace(low+1, high+1, 100)[:, None]-1
    to_gp_space = np.log1p
  else:
    X_gpr = np.linspace(low, high, 100)[:, None]
    to_gp_space = lambda x: x
  low, high = None, None
  kernel_scale=[.5, 2e0]
  noise_scale=[3e-3,1e-1]
  kernel = sklearn.gaussian_process.kernels.RBF(
      length_scale_bounds=kernel_scale
    ) + sklearn.gaussian_process.kernels.WhiteKernel(
        noise_level_bounds=noise_scale
    ) + sklearn.gaussian_process.kernels.ConstantKernel(
        constant_value_bounds=[0.4,0.85]
    )
  gpr = sklearn.gaussian_process.GaussianProcessRegressor(
    kernel=kernel, random_state=0, n_restarts_optimizer=10)
  kernel = sklearn.gaussian_process.kernels.RBF(
      length_scale_bounds=[1,2]
    ) + sklearn.gaussian_process.kernels.ConstantKernel(
      constant_value_bounds=noise_scale
    )
  gpr2 = sklearn.gaussian_process.GaussianProcessRegressor(
    kernel=kernel, random_state=0, n_restarts_optimizer=10)
  fig, axs = plt.subplots(2, sharex=True)
  X_max = np.nan
  plt.ion()
  for iter in range(budget):
    params = setups.iloc[iter % setups.shape[0]]
    try:
      metric, epoch_loss_history = experiment.train(tqdm_prefix=None, **params.to_dict())
      experiment.clear_cache()
    except KeyboardInterrupt:
      print("KeyboardInterrupt")
      break
    steps = core.metrics.argbest(epoch_loss_history, args.task)
    stats.loc[stats.shape[0]] = params[K], metric, steps, np.nan, np.nan
    plt.sca(axs[1])
    plt.scatter(stats.X, stats.S, color="lightgray")
    plt.sca(axs[0])
    plt.scatter(stats.X, stats.Y, color="black")

    # Define the kernel: RBF for smooth changes, WhiteKernel for noise level
    gpr.fit(to_gp_space(stats.X.values)[:, None], stats.Y)
    Y_gpr, S_gpr = gpr.predict(to_gp_space(X_gpr), return_std=True)
    stats.E = gpr.predict(to_gp_space(stats.X.values)[:, None])
    stats.V = (stats.Y - stats.E)**2
    gpr2.fit(to_gp_space(stats.X.values)[:, None], stats.V)
    print(f"kernel[{iter+1}/{budget}]\n={gpr.kernel_}\n={gpr2.kernel_}")
    V_gpr = gpr2.predict(to_gp_space(X_gpr))
    V_gpr = np.maximum(V_gpr, 0)

    X_max = X_gpr[np.argmax(Y_gpr + S_gpr)]
    if high is None:
      high, = plt.plot(X_gpr, Y_gpr + 2 * S_gpr, color='darkgray', zorder=-10, linestyle=":", linewidth=3)
      low, = plt.plot(X_gpr, Y_gpr, color="darkgray", linestyle="--", linewidth=5, alpha=0.5)
      var, = plt.plot(X_gpr, Y_gpr + np.sqrt(V_gpr))
      argmax = plt.axvline(X_max, color="gray", linestyle=":")
      if scale == "log":
        plt.xscale('log')
      plt.ylim([.4, .85])
    else:
      high.set_ydata(Y_gpr + 2 * S_gpr)
      low.set_ydata(Y_gpr)
      var.set_ydata(Y_gpr + 2 * np.sqrt(V_gpr))
      argmax.set_xdata(X_max)
    plt.pause(0.1)
  plt.ioff()
  print(stats.groupby("X").mean().sort_index())
  plt.show()