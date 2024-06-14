import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import register_scale
from scipy.stats import norm

from plot.probit import ProbitScale

def plot_palette(roc):
  fig, axs = plt.subplots(1, 2, figsize=(12, 5))
  plt.sca(axs[0])
  colors = plot_roc(roc)
  plt.sca(axs[1])
  plot_brier(roc, colors)

def plot_roc(roc):
  register_scale(ProbitScale)
  auc_empirical = np.trapz(roc.tpr_empirical, roc.fpr_empirical)
  auc_convex = np.trapz(roc.tpr_convex, roc.fpr_convex)
  auc_logistic = np.trapz(roc.tpr_logistic, roc.fpr_logistic)
  auc_hat = np.trapz(roc.tpr_hat, roc.fpr_hat)

  colors = {}
  colors["empirical"] = plt.plot(roc.fpr_empirical, roc.tpr_empirical, alpha=0.5, label=f"empirical: {100 * auc_empirical:.1f}%")[0].get_color()
  colors["convex"] = plt.plot(roc.fpr_convex, roc.tpr_convex, alpha=0.8, label=f"convex: {100 * auc_convex:.1f}%")[0].get_color()
  colors["logistic"] = plt.plot(roc.fpr_logistic, roc.tpr_logistic, alpha=0.8, label=f"logistic: {100 * auc_logistic:.1f}%")[0].get_color()
  colors["hat"] = plt.plot(roc.fpr_hat, roc.tpr_hat, alpha=0.8, label=f"hat: {100 * auc_hat:.1f}%")[0].get_color()

  low_z = norm.ppf(roc.tpr_empirical.iloc[1])
  high_z = norm.ppf(roc.tpr_empirical.iloc[-2])
  fpr_z = np.linspace(low_z, high_z, 100)

  for i in range(5):
    plt.plot(norm.cdf(fpr_z), norm.cdf(fpr_z + i), color="lightgray", linestyle=':', zorder=-10)

  skew = roc.targets.mean()
  for p in [0.1, 0.5]:
    frac = norm.cdf(np.linspace(low_z, high_z, 10000))
    fpr = frac / (1 - skew) * p
    tpr = (1 - frac) / skew * p
    idx = (fpr < norm.cdf(high_z)) * (tpr < norm.cdf(high_z)) * (fpr > norm.cdf(low_z)) * (tpr > norm.cdf(low_z))
    plt.plot(fpr[idx], tpr[idx], color="lightgray", linestyle=':', zorder=-10)

  plt.xlabel('fpr')
  plt.xscale('probit')
  plt.xlim([1e-2, 1 - 1e-2])
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()

  plt.ylabel('tpr')
  plt.yscale('probit')
  plt.ylim([1e-2, 1 - 1e-2])

  plt.title('ROC')
  plt.gca().set_aspect('equal')
  plt.legend()
  return colors

def plot_brier(roc, colors):
  brier_empirical = np.trapz(roc.cost_empirical, -roc.y_hat)
  brier_convex = np.trapz(roc.cost_convex, -roc.y_convex)
  brier_logistic = np.trapz(roc.cost_logistic, -roc.y_logistic)

  optimal = np.minimum(roc.y_hat, 1 - roc.y_hat)
  plt.plot(roc.y_hat, optimal - roc.cost_empirical, color=colors["empirical"], label="empirical")
  optimal = np.minimum(roc.y_logistic, 1 - roc.y_logistic)
  plt.plot(roc.y_logistic, optimal - roc.cost_logistic, color=colors["logistic"], label="logistic")
  optimal = np.minimum(roc.y_convex, 1 - roc.y_convex)
  plt.plot(roc.y_convex, optimal - roc.cost_convex, color=colors["convex"], label="convex")

  plt.xlabel('skew')
  plt.xscale('logit')

  plt.ylabel('cost (negative delta from optimal)')
  plt.yscale('log')
  plt.ylim([-1, 0])

  plt.title('Brier')
  plt.gca().invert_yaxis()
  plt.legend()