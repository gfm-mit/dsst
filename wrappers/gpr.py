import numpy as np
import sklearn.gaussian_process
import matplotlib.pyplot as plt


class GPR:
  def __init__(self, args, K, scale, budget, min, max):
    self.args = args
    self.K = K
    self.scale = scale
    self.budget = budget
    kernel_scale=[.5, 2e0]
    noise_scale=[3e-3,1e-1]
    k1 = sklearn.gaussian_process.kernels.RationalQuadratic(length_scale_bounds=kernel_scale
      ) + sklearn.gaussian_process.kernels.WhiteKernel(noise_level_bounds=noise_scale)
    self.gpr = sklearn.gaussian_process.GaussianProcessRegressor(
      kernel=k1, random_state=0, normalize_y=True, n_restarts_optimizer=10)
    k2 = sklearn.gaussian_process.kernels.RBF(length_scale_bounds=[1,2])
    self.gpr2 = sklearn.gaussian_process.GaussianProcessRegressor(
      kernel=k2, random_state=0, n_restarts_optimizer=10)

    if scale == "log":
      self.to_gp_space = np.log
      self.X = np.geomspace(min, max, 100)[:, None]
    elif scale == "log1p":
      self.to_gp_space = np.log1p
      self.X = np.geomspace(min+1, max+1, 100)[:, None]-1
    else:
      self.to_gp_space = lambda x: x
      self.X = np.linspace(min, max, 100)[:, None]
    self.interactive = False
 
  def fit(self, stats):
    self.gpr.fit(self.to_gp_space(stats.X.values)[:, None], stats.Y)
    self.Y, self.S = self.gpr.predict(self.to_gp_space(self.X), return_std=True)
    E = self.gpr.predict(self.to_gp_space(stats.X.values)[:, None])
    V = np.log((stats.Y - E)**2+1e-8)
    self.gpr2.fit(self.to_gp_space(stats.X.values)[:, None], V)
    self.V = np.exp(self.gpr2.predict(self.to_gp_space(self.X)))
    print(f"mean={self.gpr.kernel_}\nvar={self.gpr2.kernel_}")

  def make_plot(self, axs):
    plt.sca(axs[0])
    self.highest, = plt.plot(self.X, self.Y + 2 * self.S + 2 * np.exp(self.V/2))
    self.high, = plt.plot(self.X, self.Y + 2 * np.exp(self.V/2))
    self.med, = plt.plot(self.X, self.Y)
    self.low, = plt.plot(self.X, self.Y - 2 * np.exp(self.V/2))
    if self.scale == "log":
      plt.xscale('log')
    elif self.scale == "log1p":
      plt.xscale('symlog', linthresh=1)
    else:
      plt.xscale('linear')
    plt.ylim([0, 3])
  
  def update_plot(self, axs):
    if not self.interactive:
      self.interactive = True
      return self.make_plot(axs)
    plt.sca(axs[0])
    self.highest.set_ydata(self.Y + 2 * self.S + 2 * np.exp(self.V/2))
    self.high.set_ydata(self.Y + 2 * np.exp(self.V/2))
    self.med.set_ydata(self.Y)
    self.low.set_ydata(self.Y - 2 * np.exp(self.V/2))
  
  def scatter(self, stats, axs):
    plt.sca(axs[0])
    plt.scatter(stats.X, stats.Y, color="black", s=100, alpha=0.5)
    plt.sca(axs[1])
    plt.scatter(stats.X, stats.S, color="black", s=10)