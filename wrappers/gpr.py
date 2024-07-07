import warnings

import numpy as np
import sklearn.gaussian_process
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt


class GPR:
  def __init__(self, K, scale, budget, min, max, sd=None):
    self.K = K
    self.scale = scale
    self.budget = budget
    kernel = sklearn.gaussian_process.kernels.RationalQuadratic(
      length_scale=0.2,
      length_scale_bounds='fixed',
      alpha_bounds=[1e-1, 1e1],
      ) + sklearn.gaussian_process.kernels.WhiteKernel()
    self.mean_gpr = sklearn.gaussian_process.GaussianProcessRegressor(
      kernel=kernel, random_state=0, normalize_y=True, n_restarts_optimizer=10)

    if scale == "log":
      self.to_gp_space = np.log10
      self.X = np.geomspace(min, max, 100)[:, None]
    elif scale == "log1p":
      self.to_gp_space = lambda x: np.log10(1+x)
      self.X = np.geomspace(min+1, max+1, 100)[:, None]-1
    else:
      self.to_gp_space = lambda x: x
      self.X = np.linspace(min, max, 100)[:, None]
    self.interactive = False
 
  def fit(self, stats):
    old_format = warnings.formatwarning
    warnings.formatwarning = lambda message, category, filename, lineno, line: f"MEAN-GPR: {message}\n"

    self.mean_gpr.fit(self.to_gp_space(stats.X.values)[:, None], stats.Y)
    self.Y, self.S = self.mean_gpr.predict(self.to_gp_space(self.X), return_std=True)
    print(f"{self.mean_gpr.kernel_=}")

    XY = self.mean_gpr.kernel_.k1(
      self.to_gp_space(stats.X.values[:, None]),
      self.to_gp_space(self.X),
      )
    XY = XY.sum(axis=0)
    self.S2 = self.S / np.sqrt(XY)

    warnings.formatwarning = old_format

  def make_plot(self, axs):
    plt.sca(axs[0])
    self.band = plt.fill_between(self.X[:, 0], self.Y + 2 * self.S, self.Y - 2 * self.S, alpha=0.1, zorder=-20)
    self.band2 = plt.fill_between(self.X[:, 0], self.Y + 2 * self.S2, self.Y - 2 * self.S2, alpha=0.5, zorder=+10)
    if self.scale == "log":
      plt.xscale('log')
    elif self.scale == "log1p":
      plt.xscale('symlog', linthresh=1)
    else:
      plt.xscale('linear')
    plt.ylim([1.0, 2])
  
  def update_plot(self, axs):
    if not self.interactive:
      self.interactive = True
      return self.make_plot(axs)
    plt.sca(axs[0])
    self.band.remove()
    self.band = plt.fill_between(self.X[:, 0], self.Y + 2 * self.S, self.Y - 2 * self.S, alpha=0.1, zorder=-20)
    self.band2.remove()
    self.band2 = plt.fill_between(self.X[:, 0], self.Y + 2 * self.S2, self.Y - 2 * self.S2, alpha=0.5, zorder=+10)
  
  def scatter(self, stats, axs):
    plt.sca(axs[0])
    plt.scatter(stats.X, stats.Y, color="black", s=100, alpha=0.1)
    plt.sca(axs[1])
    plt.scatter(stats.X, stats.S, color="black", s=10)