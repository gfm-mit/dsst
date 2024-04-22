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
  plt.plot(convex.fpr, convex.tpr, linewidth=3, alpha=0.2, color=color)
  plt.ylabel('TPR')
  plt.xlabel('FPR')
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()

  plt.sca(axs[1])
  plt.title("Operating Characteristics\n(Q-Q plot)")
  plt.scatter(roc.fpr_literal, roc.tpr_literal, alpha=0.5, color=color)
  plt.plot(interpolated_smooth.fpr, interpolated_smooth.tpr, linewidth=3, alpha=0.2, color=color)

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
  plt.title('rotated ROC curve')
  plus = convex.tpr + convex.fpr
  minus = convex.tpr - convex.fpr
  plt.plot(plus, minus, linewidth=3, alpha=0.2, color=color)
  plus = roc.tpr_literal + roc.fpr_literal
  minus = roc.tpr_literal - roc.fpr_literal
  plt.scatter(plus, minus, s=1)

  plt.plot([0, 1, 2], [0, 1, 0], linestyle='--', color="lightgray", zorder=-10)
  plt.ylabel('TPR - FPR')
  plt.xlabel('TPR + FPR')
  axs[2].set_aspect('equal')
  return color

def plot_9_types(predicted, labels, axs):
  register_scale(ProbitScale)
  register_scale(LogOneMinusXScale)
  roc = get_roc(predicted, labels)
  convex = get_roc_convex_hull(roc.shape[0], roc.fpr_literal.values, roc.tpr_literal.values)
  interpolated_smooth = get_roc_interpolated_convex_hull(convex.fpr, convex.tpr)
  color = plot_roc(roc, convex, interpolated_smooth, axs)

  return
  project(convex.fpr, convex.tpr, axs)

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
  plt.ylabel('Number Needed to Evaluate')

def project(fpr, tpr, axs):
  slopes = np.diff(tpr) / np.diff(fpr)
  slopes[np.diff(fpr) == 0] = np.inf
  slopes[np.diff(tpr) == 0] = 0
  lrt = np.geomspace(1e-2, 1e2, 100)
  idx = np.searchsorted(slopes, lrt)
  y = tpr[idx] - fpr[idx] * lrt
  y0 = np.maximum(0, 1 - 1 * lrt)
  ymax = np.ones_like(y)
  ymin = -lrt
  dy = (y - ymin) / (ymax - ymin)
  dy0 = (y0 - ymin) / (ymax - ymin)
  
  plt.sca(axs[4])
  plt.plot(lrt, dy)
  plt.plot(lrt, dy0, color="lightgray", linestyle="--", zorder=-10)
  plt.xscale("log")
  plt.xlabel('Likelihood Ratio')
  plt.ylabel('Value')
  plt.yticks([0,1], ["Completely\nWrong", "Completely\nRight"])

  plt.sca(axs[5])
  plt.plot(lrt, dy - dy0)
  plt.plot(lrt, 1 - dy0, color="lightgray", linestyle="--", zorder=-10)
  plt.xscale("log")
  plt.ylabel('Δ Value')
  plt.xlabel('Likelihood Ratio')

  plt.sca(axs[8])
  plt.plot(lrt, tpr[idx] - fpr[idx])
  plt.xscale("log")
  plt.ylabel('Δ Equivalent TP\nwith zero FP')
  plt.xlabel('Likelihood Ratio\n(at Threshold, evaluation uses 1)')

  plt.sca(axs[3])
  color = plt.plot(lrt, fpr[idx])[0].get_color()
  plt.plot(lrt, tpr[idx], color=color, dashes=[1,1])
  plt.legend(["FPR", "TPR"])
  plt.xlabel('Likelihood Ratio')
  plt.xscale("log")
