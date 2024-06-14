import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import register_scale
from matplotlib import ticker
from scipy.stats import norm

from plot.probit import ProbitScale

def plot_palette(roc, axs=None, label_alternatives=False):
  if axs is None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
  plt.sca(axs[0])
  color = plot_roc(roc, label_alternatives=label_alternatives)
  plt.sca(axs[1])
  plot_brier(roc, color, label_alternatives=label_alternatives)
  return axs

def plot_roc(roc, label_alternatives=False):
  register_scale(ProbitScale)
  auc_empirical = np.trapz(roc.tpr_empirical, roc.fpr_empirical)
  auc_convex = np.trapz(roc.tpr_convex, roc.fpr_convex)
  auc_logistic = np.trapz(roc.tpr_logistic, roc.fpr_logistic)
  auc_hat = np.trapz(roc.tpr_hat, roc.fpr_hat)

  color = plt.plot(roc.fpr_empirical, roc.tpr_empirical, label=f"empirical: {100 * auc_empirical:.1f}%" if label_alternatives else f"AUC: {100 * auc_empirical:.1f}%")[0].get_color()
  plt.plot(roc.fpr_convex, roc.tpr_convex, alpha=0.5, color=color, linewidth=5, zorder=-5, label=f"convex: {100 * auc_convex:.1f}%" if label_alternatives else None)
  plt.plot(roc.fpr_logistic, roc.tpr_logistic, alpha=0.25, linestyle="--", color=color, label=f"logistic: {100 * auc_logistic:.1f}%" if label_alternatives else None)
  #colors["hat"] = plt.plot(roc.fpr_hat, roc.tpr_hat, alpha=0.8, label=f"hat: {100 * auc_hat:.1f}%")[0].get_color()

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
  return color

def plot_brier(roc, color, label_alternatives=False):
  brier_empirical = np.trapz(roc.cost_empirical, -roc.y_hat)
  brier_convex = np.trapz(roc.cost_convex, -roc.y_convex)
  brier_logistic = np.trapz(roc.cost_logistic, -roc.y_logistic)

  base_rate_only = np.minimum(roc.y_hat, 1 - roc.y_hat)
  plt.plot(roc.y_hat, roc.cost_empirical - base_rate_only, color=color, label=f"empirical: {400 * brier_empirical - 100:.1f}%" if label_alternatives else f"Brier(ish): {400 * brier_empirical - 100:.1f}%")
  base_rate_only = np.minimum(roc.y_convex, 1 - roc.y_convex)
  plt.plot(roc.y_convex, roc.cost_convex - base_rate_only, alpha=0.5, color=color, linewidth=5, zorder=-5, label=f"convex: {400 * brier_convex - 100:.1f}%" if label_alternatives else None)
  base_rate_only = np.minimum(roc.y_logistic, 1 - roc.y_logistic)
  plt.plot(roc.y_logistic, roc.cost_logistic - base_rate_only, alpha=0.25, linestyle="--", color=color, label=f"logistic: {400 * brier_logistic - 100:.1f}%" if label_alternatives else None)
  #base_rate_only = np.minimum(roc.y_hat, 1 - roc.y_hat)
  #plt.plot(roc.y_hat, base_rate_only - roc.cost_hat, color=colors["convex"], label="convex")

  plt.xlabel('skew')
  plt.xscale('logit')
  plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
  plt.gca().xaxis.set_minor_formatter(lambda x, pos: "")
  plt.xlim([1e-1, 1 - 1e-1])

  plt.ylabel('cost (delta from only using base rates)')
  plt.yscale('symlog', linthresh=1e-2)
  plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
  #plt.ylim([-1, 0])

  plt.title('Brier')
  plt.legend()