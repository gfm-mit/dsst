import warnings

import numpy as np
import pandas as pd
import sklearn.gaussian_process
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt

import core.metrics


class GPR:
  def __init__(self, K, scale, budget, task="next_token", sigma=0.1, **kwargs):
    print(f"{kwargs=}")
    self.K = K
    self.scale = scale
    self.budget = budget
    self.task=task
    kernel = sklearn.gaussian_process.kernels.RationalQuadratic(
      length_scale=sigma,
      length_scale_bounds='fixed',
      ) + sklearn.gaussian_process.kernels.WhiteKernel()
    self.gpr = sklearn.gaussian_process.GaussianProcessRegressor(
      kernel=kernel, random_state=0, normalize_y=True, n_restarts_optimizer=10)

    if scale == "log":
      self.to_gp_space = np.log10
      self.from_gp_space = lambda x: 10 ** x
    elif scale == "log1p":
      self.to_gp_space = lambda x: np.log10(1+x)
      self.from_gp_space = lambda x: 10 ** x - 1
    else:
      self.to_gp_space = lambda x: x
      self.from_gp_space = lambda x: x
  
  def gp_space_plus(self, x, dx):
    return self.from_gp_space(self.to_gp_space(x) + dx)

  def fit_predict(self, stats, targets):
    if stats.shape[0] == 0:
      targets["Y"] = 0
      targets["S"] = 0
      targets["S_mu"] = np.exp(
        np.linspace(-1, 1, num=targets.shape[0])**2
      )
      return targets, 0
    self.fit(stats)
    return self.predict(targets)
 
  def fit(self, stats):
    old_format = warnings.formatwarning
    warnings.formatwarning = lambda message, category, filename, lineno, line: f"MEAN-GPR: {message}\n"

    self.gpr.fit(self.to_gp_space(stats.X.values)[:, None], stats.Y)
    print(f"{self.gpr.kernel_=}")
    warnings.formatwarning = old_format
  
  def get_default_targets(self, min, max, **kwargs):
    print(f"{kwargs=}")
    if self.scale == "log":
      X = np.geomspace(min, max, 100)
    elif self.scale == "log1p":
      X = np.geomspace(min+1, max+1, 100)-1
    else:
      X = np.linspace(min, max, 100)
    return pd.DataFrame(X, columns=["X"])
  
  def predict(self, targets):
    assert targets is not None
    targets["Y"], targets["S"] = self.gpr.predict(self.to_gp_space(targets.X.values[:, None]), return_std=True)

    XY = self.gpr.kernel_.k1(
      self.gpr.X_train_,
      self.to_gp_space(targets.X.values[:, None]),
      )
    XY = XY.sum(axis=0)

    total_N = self.gpr.X_train_.shape[0]

    targets["S_mu"] = targets.S / np.sqrt(XY)
    if total_N == 0:
      targets.S_mu = np.exp(
        np.linspace(-1, 1, num=targets.shape[0])**2
      )
    elif total_N == 1:
      D = (self.to_gp_space(targets.X) - self.to_gp_space(self.gpr.X_train_[0]))**2
      D /= D.max()
      targets.S_mu = np.exp(D)
    else:
      targets.S_mu *= np.sqrt(4 * np.log(total_N-1) / XY)

    if self.task == "next_token":
      best_idx = np.argmin(targets.Y - 2 * targets.S_mu)
    else:
      best_idx = np.argmax(targets.Y + 2 * targets.S_mu)
    return targets, best_idx

  def make_plot(self, targets, best_idx, axs):
    if axs is None:
      fig, axs = plt.subplots(2, sharex=True)
    plt.sca(axs[0])
    self.band = plt.fill_between(targets.X, targets.Y + 2 * targets.S, targets.Y - 2 * targets.S, alpha=0.1, zorder=-10)
    self.band2 = plt.fill_between(targets.X, targets.Y + 2 * targets.S_mu, targets.Y - 2 * targets.S_mu, alpha=0.5, zorder=10)
    self.vline = plt.axvline(x=targets.X.iloc[best_idx], color="lightgray", linewidth=2, linestyle=':', zorder=-20)
    sigma_x = self.gpr.kernel_.k1.length_scale
    self.band3 = plt.axvspan(
      self.gp_space_plus(targets.X.iloc[best_idx],-sigma_x),
      self.gp_space_plus(targets.X.iloc[best_idx],+sigma_x),
      color="limegreen", alpha=0.05, zorder=-30)
    if self.scale == "log":
      plt.xscale('log')
    elif self.scale == "log1p":
      plt.xscale('symlog', linthresh=1)
    else:
      plt.xscale('linear')
    return axs
  
  def update_plot(self, targets, best_idx, axs):
    if axs is None:
      return self.make_plot(targets, best_idx, axs)
    plt.sca(axs[0])
    self.band.remove()
    self.band = plt.fill_between(targets.X, targets.Y + 2 * targets.S, targets.Y - 2 * targets.S, alpha=0.1, zorder=-10)
    self.band2.remove()
    self.band2 = plt.fill_between(targets.X, targets.Y + 2 * targets.S_mu, targets.Y - 2 * targets.S_mu, alpha=0.5, zorder=10)
    self.vline.set_xdata(targets.X[best_idx])
    sigma_x = self.gpr.kernel_.k1.length_scale
    self.band3.set_x(self.gp_space_plus(targets.X.iloc[best_idx],-sigma_x))
    self.band3.set_width(
      self.gp_space_plus(targets.X.iloc[best_idx],+sigma_x)
      - self.gp_space_plus(targets.X.iloc[best_idx],-sigma_x))
    return axs
  
  def scatter(self, stats, axs):
    plt.sca(axs[0])
    plt.scatter(stats.X, stats.Y, color="black", s=100, alpha=0.1)
    plt.sca(axs[1])
    plt.scatter(stats.X, stats.S, color="black", s=10)