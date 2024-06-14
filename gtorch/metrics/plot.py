import numpy as np
import matplotlib.pyplot as plt


def plot_roc(roc):
  auc_empirical = np.trapz(roc.tpr_empirical, roc.fpr_empirical)
  auc_convex = np.trapz(roc.tpr_convex, roc.fpr_convex)
  auc_logistic = np.trapz(roc.tpr_logistic, roc.fpr_logistic)
  auc_hat = np.trapz(roc.tpr_hat, roc.fpr_hat)

  fig, axs = plt.subplots(2, 2)
  axs = axs.flatten()
  plt.sca(axs[0])
  empirical_color = plt.plot(roc.fpr_empirical, roc.tpr_empirical, alpha=0.5, label=f"empirical: {100 * auc_empirical:.1f}%")[0].get_color()
  convex_color = plt.plot(roc.fpr_convex, roc.tpr_convex, alpha=0.8, label=f"convex: {100 * auc_convex:.1f}%")[0].get_color()
  logistic_color = plt.plot(roc.fpr_logistic, roc.tpr_logistic, alpha=0.8, label=f"logistic: {100 * auc_logistic:.1f}%")[0].get_color()
  hat_color = plt.plot(roc.fpr_hat, roc.tpr_hat, alpha=0.8, label=f"hat: {100 * auc_hat:.1f}%")[0].get_color()
  plt.legend()
  plt.gca().set_aspect('equal')
  plt.xlabel('fpr')
  plt.ylabel('tpr')

  plt.sca(axs[1])
  plt.scatter(roc.y_hat, roc.y_convex, label="hat", color=hat_color)
  plt.scatter(roc.y_logistic, roc.y_convex, label="logistic", color=logistic_color)
  plt.legend()
  plt.gca().set_aspect('equal')
  plt.xlabel('y_guess')
  plt.ylabel('y_convex')

  brier_empirical = np.trapz(roc.cost_empirical, -roc.y_hat)
  brier_convex = np.trapz(roc.cost_convex, -roc.y_hat)
  brier_logistic = np.trapz(roc.cost_logistic, -roc.y_hat)

  plt.sca(axs[2])
  plt.plot(roc.y_hat, roc.cost_empirical, color=empirical_color, label=f"empirical: {100 * brier_empirical:.1f}")
  plt.plot(roc.y_logistic, roc.cost_logistic, color=logistic_color, label=f"logistic: {100 * brier_logistic:.1f}")
  plt.plot(roc.y_convex, roc.cost_convex, color=convex_color, label=f"convex: {100 * brier_convex:.1f}")
  plt.legend()
  plt.xlabel('skew')
  plt.ylabel('cost')