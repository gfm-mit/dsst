import numpy as np
import sklearn.gaussian_process
import matplotlib.pyplot as plt


class GPR:
  def __init__(self, K, scale, budget, min, max, sd=0.01):
    self.K = K
    self.scale = scale
    self.budget = budget
    self.sd = sd
    k1 = sklearn.gaussian_process.kernels.RationalQuadratic(length_scale_bounds=[.5, 2e0]
      ) + sklearn.gaussian_process.kernels.WhiteKernel(noise_level_bounds=[3e-3,1])
    self.mean_gpr = sklearn.gaussian_process.GaussianProcessRegressor(
      kernel=k1, random_state=0, normalize_y=True, n_restarts_optimizer=10)
    k2 = sklearn.gaussian_process.kernels.RBF(length_scale_bounds=[1,2]
      ) + sklearn.gaussian_process.kernels.WhiteKernel(
      noise_level_bounds=[1e-2,3e-1]
      )
    self.var_gpr = sklearn.gaussian_process.GaussianProcessRegressor(
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
    try:
      self.mean_gpr.fit(self.to_gp_space(stats.X.values)[:, None], stats.Y)
    except sklearn.gaussian_process.kernels.ConvergenceWarning as w:
      print(w)
    self.Y, self.S = self.mean_gpr.predict(self.to_gp_space(self.X), return_std=True)
    E = self.mean_gpr.predict(self.to_gp_space(stats.X.values)[:, None])
    V = (stats.Y - E)**2
    LV = np.log(V + 1e-12) - np.log(self.sd)
    self.var_gpr.fit(self.to_gp_space(stats.X.values)[:, None], LV)
    self.LV = self.var_gpr.predict(self.to_gp_space(self.X))
    self.S2 = np.sqrt(self.sd * np.exp(self.LV))
    print(self.S2)
    print(f"mean={self.mean_gpr.kernel_}\nvar={self.var_gpr.kernel_}")

  def make_plot(self, axs):
    plt.sca(axs[0])
    #self.med, = plt.plot(self.X, self.Y)
    self.band = plt.fill_between(self.X[:, 0], self.Y + 2 * self.S, self.Y - 2 * self.S, alpha=0.3, zorder=-100)
    self.high, = plt.plot(self.X, self.Y + 2 * self.S2)
    self.low, = plt.plot(self.X, self.Y - 2 * self.S2)
    self.high.set_color(self.low.get_color())
    if self.scale == "log":
      plt.xscale('log')
    elif self.scale == "log1p":
      plt.xscale('symlog', linthresh=1)
    else:
      plt.xscale('linear')
    plt.ylim([0.5, 2])
  
  def update_plot(self, axs):
    if not self.interactive:
      self.interactive = True
      return self.make_plot(axs)
    plt.sca(axs[0])
    #self.med.set_ydata(self.Y)
    self.band.remove()
    self.band = plt.fill_between(self.X[:, 0], self.Y + 2 * self.S, self.Y - 2 * self.S, alpha=0.1, zorder=-100)
    self.high.set_ydata(self.Y + 2 * self.S2)
    self.low.set_ydata(self.Y - 2 * self.S2)
  
  def scatter(self, stats, axs):
    plt.sca(axs[0])
    plt.scatter(stats.X, stats.Y, color="black", s=100, alpha=0.1)
    plt.sca(axs[1])
    plt.scatter(stats.X, stats.S, color="black", s=10)