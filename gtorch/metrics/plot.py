import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import register_scale
from scipy.stats import norm

from plot.probit import ProbitScale

def plot_roc(roc):
  register_scale(ProbitScale)
  auc_empirical = np.trapz(roc.tpr_empirical, roc.fpr_empirical)
  auc_convex = np.trapz(roc.tpr_convex, roc.fpr_convex)
  auc_logistic = np.trapz(roc.tpr_logistic, roc.fpr_logistic)
  auc_hat = np.trapz(roc.tpr_hat, roc.fpr_hat)

  fig, axs = plt.subplots(1, 2, figsize=(12, 5))
  axs = axs.flatten()
  plt.sca(axs[0])
  empirical_color = plt.plot(roc.fpr_empirical, roc.tpr_empirical, alpha=0.5, label=f"empirical: {100 * auc_empirical:.1f}%")[0].get_color()
  convex_color = plt.plot(roc.fpr_convex, roc.tpr_convex, alpha=0.8, label=f"convex: {100 * auc_convex:.1f}%")[0].get_color()
  logistic_color = plt.plot(roc.fpr_logistic, roc.tpr_logistic, alpha=0.8, label=f"logistic: {100 * auc_logistic:.1f}%")[0].get_color()
  hat_color = plt.plot(roc.fpr_hat, roc.tpr_hat, alpha=0.8, label=f"hat: {100 * auc_hat:.1f}%")[0].get_color()

  low_z = norm.ppf(roc.tpr_empirical.iloc[1])
  high_z = norm.ppf(roc.tpr_empirical.iloc[-2])
  fpr_z = np.linspace(low_z, high_z, 100)
  print(low_z, high_z)
  for i in range(5):
    plt.plot(norm.cdf(fpr_z), norm.cdf(fpr_z + i), color="lightgray", linestyle=':', zorder=-10)
  skew = roc.targets.mean()
  for p in [0.1, 0.5]:
    frac = norm.cdf(np.linspace(low_z, high_z, 10000))
    fpr = frac / (1 - skew) * p
    tpr = (1 - frac) / skew * p
    idx = (fpr < norm.cdf(high_z)) * (tpr < norm.cdf(high_z)) * (fpr > norm.cdf(low_z)) * (tpr > norm.cdf(low_z))
    plt.plot(fpr[idx], tpr[idx], color="lightgray", linestyle=':', zorder=-10)
  plt.xlim([1e-2, 1 - 1e-2])
  plt.ylim([1e-2, 1 - 1e-2])
  plt.legend()
  plt.xlabel('fpr')
  plt.ylabel('tpr')
  plt.xscale('probit')
  plt.yscale('probit')
  plt.gca().set_aspect('equal')
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.title('ROC')

  #plt.sca(axs[1])
  #plt.scatter(roc.y_hat, roc.y_convex, label="hat", color=hat_color)
  #plt.scatter(roc.y_logistic, roc.y_convex, label="logistic", color=logistic_color)
  #plt.legend()
  #plt.gca().set_aspect('equal')
  #plt.xlabel('y_guess')
  #plt.ylabel('y_convex')

  brier_empirical = np.trapz(roc.cost_empirical, -roc.y_hat)
  brier_convex = np.trapz(roc.cost_convex, -roc.y_convex)
  brier_logistic = np.trapz(roc.cost_logistic, -roc.y_logistic)

  #plt.sca(axs[2])
  #plt.plot(roc.y_hat, roc.cost_empirical, color=empirical_color, label=f"empirical: {100 * brier_empirical:.1f}%")
  #plt.plot(roc.y_logistic, roc.cost_logistic, color=logistic_color, label=f"logistic: {100 * brier_logistic:.1f}%")
  #plt.plot(roc.y_convex, roc.cost_convex, color=convex_color, label=f"convex: {100 * brier_convex:.1f}%")
  #plt.legend()
  #plt.xlabel('skew')
  #plt.ylabel('cost')
  #plt.gca().set_aspect('equal')

  plt.sca(axs[1])
  optimal = np.minimum(roc.y_hat, 1 - roc.y_hat)
  plt.plot(roc.y_hat, optimal - roc.cost_empirical, color=empirical_color, label="empirical")
  optimal = np.minimum(roc.y_logistic, 1 - roc.y_logistic)
  plt.plot(roc.y_logistic, optimal - roc.cost_logistic, color=logistic_color, label="logistic")
  optimal = np.minimum(roc.y_convex, 1 - roc.y_convex)
  plt.plot(roc.y_convex, optimal - roc.cost_convex, color=convex_color, label="convex")
  plt.xscale('logit')
  plt.yscale('log')
  plt.legend()
  plt.ylim([-1, 0])
  plt.xlabel('skew')
  plt.ylabel('cost (negative delta from optimal)')
  plt.title('Brier')
  plt.gca().invert_yaxis()