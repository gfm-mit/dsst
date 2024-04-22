import pandas as pd
import matplotlib.pyplot as plt
from plot.probit import *
from plot.log_one_minus import *
from plot.roc import *
from scipy.stats import norm

def plot_roc(roc, convex, interpolated_smooth, axs):
  plt.sca(axs[0])
  plt.title("Operating Characteristics")
  color = plt.scatter(roc.fpr_literal, roc.tpr_literal, s=1).get_facecolor()[0]
  plt.scatter(convex.fpr[1:-1], convex.tpr[1:-1], s=100, alpha=0.2, color=color)
  plt.plot(convex.fpr, convex.tpr, linewidth=3, alpha=0.4, color=color)
  plt.ylabel('TPR')
  plt.xlabel('FPR')
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()

  plt.sca(axs[1])
  plt.title("Operating Characteristics\n(Q-Q plot)")
  plt.scatter(roc.fpr_literal, roc.tpr_literal, s=1, alpha=0.7, color=color)
  plt.scatter(convex.fpr[1:-1], convex.tpr[1:-1], s=100, alpha=0.2, color=color)
  plt.plot(interpolated_smooth.fpr, interpolated_smooth.tpr, linewidth=3, alpha=0.4, color=color)

  cz0 = 1e-3
  z0 = norm.ppf(cz0)
  cz1 = 1-1e-3
  z1 = norm.ppf(cz1)

  plt.ylabel("True Positive Rate")
  plt.yscale('probit')
  plt.ylim([cz0, cz1])

  plt.xlabel("False Positive Rate")
  plt.xscale('probit')
  plt.xlim([cz0, cz1])
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()

  plt.axline((cz0, cz0), (cz1, cz1), color="lightgray", linestyle = "--", zorder=-10)
  plt.axline((cz0, norm.cdf(z0+1)), (norm.cdf(z1-1), cz1), color="lightgray", linestyle = "--", zorder=-10)
  plt.axline((cz0, norm.cdf(z0+2)), (norm.cdf(z1-2), cz1), color="lightgray", linestyle = "--", zorder=-10)

  plt.sca(axs[2])
  plt.title('affine transformed\nROC curve')
  plus = convex.tpr + convex.fpr
  minus = convex.tpr - convex.fpr
  plt.plot(-plus, 1+minus, linewidth=3, alpha=0.4, color=color)
  plt.scatter(-plus[1:-1], 1+minus[1:-1], linewidth=3, s=100, alpha=0.2, color=color)
  plus = roc.tpr_literal + roc.fpr_literal
  minus = roc.tpr_literal - roc.fpr_literal
  plt.scatter(-plus, 1+minus, s=1)

  plt.plot([-0, -1, -2], [1, 2, 1], linestyle='--', color="lightgray", zorder=-10)
  plt.plot([-0, -2], [1, 1], linestyle=':', color="lightgray", zorder=-10)
  plt.ylabel('TPR + TNR (1-FPR)')
  plt.xlabel('-TPR - FPR')
  axs[2].set_aspect('equal')
  return color

def plot_eta_dollar(eta, idx, roc, convex, interpolated_smooth, axs, color):
  plt.sca(axs[3])
  plt.title('TPR + TNR vs Likelihood Ratio')
  minus = convex.tpr[idx] - convex.fpr[idx]
  #plt.plot(eta, minus, linewidth=10, alpha=0.2, color=color)
  minus[np.roll(minus, 1) == np.roll(minus, -1)] = np.nan
  plt.plot(eta, 1+minus, linewidth=2, alpha=0.4, color=color)
  minus = convex.tpr[idx] - convex.fpr[idx]
  minus[np.roll(minus, 1) != np.roll(minus, -1)] = np.nan
  plt.plot(eta, 1+minus, linewidth=10, alpha=0.2, color=color)
  plt.plot(eta, 1+np.ones_like(eta), linestyle='--', color="lightgray", zorder=-10)
  plt.plot(eta, 1+np.zeros_like(eta), linestyle=':', color="lightgray", zorder=-10)
  plt.xlabel('Likelihood Ratio')
  plt.xscale('log')
  plt.ylabel('TPR + TNR')
  
  plt.sca(axs[4])
  plt.title('Balanced Accuracy (equivalent)')
  tpr_equiv = convex.tpr[idx] - eta * convex.fpr[idx]
  tnr_equiv = 1 - convex.fpr[idx] - (1-convex.tpr[idx]) / eta
  balanced = 1 - (1-tpr_equiv) / (1+eta)
  plt.plot(eta, balanced)
  plt.xlabel('Likelihood Ratio')
  plt.xscale('log')
  plt.ylabel('balanced accuracy\n(equivalent)')
  plt.ylim([0, 1-1e-3])
  plt.yscale('log1minusx')

  plt.sca(axs[5])
  plt.title('pure TPR or TNR (equivalent)')
  flat_idx = tpr_equiv == np.roll(tpr_equiv, 1)
  plt.plot(eta[flat_idx], tpr_equiv[flat_idx], linestyle='--', color=color)
  plt.plot(eta[~flat_idx], tpr_equiv[~flat_idx], color=color)
  flat_idx = tnr_equiv == np.roll(tnr_equiv, -1)
  plt.plot(eta[flat_idx], tnr_equiv[flat_idx], linestyle='--', color=color, alpha=0.4)
  plt.plot(eta[~flat_idx], tnr_equiv[~flat_idx], color=color, alpha=0.4)
  plt.xlabel('Likelihood Ratio')
  plt.xscale('log')
  plt.ylabel('TPR')
  plt.ylim([0, 1])
  ax2 = plt.gca().twinx()
  ax2.set_ylabel('TNR', color="lightgray")
  ax2.tick_params(axis='y', color="lightgray", labelcolor="lightgray")

def plot_precision(eta, idx, roc, convex, interpolated_smooth, axs, color):
  plt.sca(axs[6])
  plt.plot(interpolated_smooth.tpr, interpolated_smooth.tpr / (interpolated_smooth.tpr + interpolated_smooth.fpr), linewidth=3, alpha=0.2, color=color)
  plt.scatter(roc.tpr_literal, roc.tpr_literal / (roc.tpr_literal + roc.fpr_literal), s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('Recall')
  plt.ylabel('Precision')

  plt.sca(axs[7])
  budget = interpolated_smooth.tpr * roc.labels.sum() + interpolated_smooth.fpr * (1-roc.labels).sum()
  nne = budget / interpolated_smooth.tpr / roc.labels.sum()
  budget = budget / roc.shape[0]
  plt.plot(budget, nne, linewidth=3, alpha=0.2, color=color)
  budget = roc.tpr_literal * roc.labels.sum() + roc.fpr_literal * (1-roc.labels).sum()
  nne = budget / roc.tpr_literal / roc.labels.sum()
  budget = budget / roc.shape[0]
  plt.scatter(budget, nne, s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('Triage Budget')
  plt.xscale('log')
  plt.xlim([1e-2, 1])
  plt.ylabel('Number\nNeeded to Evaluate')


def plot_9_types(predicted, labels, axs):
  register_scale(ProbitScale)
  register_scale(LogOneMinusXScale)
  roc = get_roc(predicted, labels)
  convex = get_roc_convex_hull(roc.shape[0], roc.fpr_literal.values, roc.tpr_literal.values)
  interpolated_smooth = get_roc_interpolated_convex_hull(convex.fpr, convex.tpr)
  color = plot_roc(roc, convex, interpolated_smooth, axs)

  eta, idx = get_slopes(convex.fpr, convex.tpr)
  plot_eta_dollar(eta, idx, roc, convex, interpolated_smooth, axs, color)
  plot_precision(eta, idx, roc, convex, interpolated_smooth, axs, color)