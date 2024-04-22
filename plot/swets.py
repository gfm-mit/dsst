import pandas as pd
import matplotlib.pyplot as plt
from plot.probit import *
from plot.log_one_minus import *
from plot.roc import *
from scipy.stats import norm
from scipy.spatial import ConvexHull

def plot_9_types(predicted, labels, axs):
  roc = get_roc(predicted, labels)
  convex = get_roc_convex_hull(roc.shape[0], roc.fpr_literal.values, roc.tpr_literal.values)
  interpolated_smooth = get_roc_interpolated_convex_hull(convex.fpr, convex.tpr)
  register_scale(ProbitScale)
  register_scale(LogOneMinusXScale)
  plt.sca(axs[4])
  color = plt.scatter(roc.fpr, roc.tpr, alpha=0.5).get_facecolor()[0]

  plt.ylim([1e-2, 1 - 1e-2])
  plt.ylabel("True Positive Rate")
  plt.yscale('probit')

  plt.xlim([1e-2, 1 - 1e-2])
  plt.xlabel("False Positive Rate")
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xscale('probit')
  plt.axline((1e-2, 1e-2), (1-1e-2, 1-1e-2), color="lightgray", linestyle = "--", zorder=-10)
  plt.axline(
    (1e-2, norm.cdf(norm.ppf(1e-2)+1)),
    (norm.cdf(norm.ppf(1-1e-2)-1), 1-1e-2),
    color="lightgray", linestyle = "--", zorder=-10)
  plt.axline(
    (1e-2, norm.cdf(norm.ppf(1e-2)+2)),
    (norm.cdf(norm.ppf(1-1e-2)-2), 1-1e-2),
    color="lightgray", linestyle = "--", zorder=-10)
  plt.suptitle("Operating Characteristics (Swets SNR Scaling)")

  plt.plot(interpolated_smooth.fpr, interpolated_smooth.tpr, linewidth=3, alpha=0.2, color=color)
  project(convex.fpr, convex.tpr, axs)

  plt.sca(axs[3])
  plt.plot(convex.fpr, convex.tpr, linewidth=3, alpha=0.2, color=color)
  plt.scatter(roc.fpr_literal, roc.tpr_literal, s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('FPR')
  plt.ylabel('TPR')

  plt.sca(axs[5])
  plt.plot(convex.fpr, convex.tpr, linewidth=3, alpha=0.2, color=color)
  plt.scatter(roc.fpr_literal, roc.tpr_literal, s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.xscale('log')

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
  
  plt.sca(axs[1])
  plt.plot(lrt, dy)
  plt.plot(lrt, dy0, color="lightgray", linestyle="--", zorder=-10)
  plt.xscale("log")
  plt.xlabel('Likelihood Ratio')
  plt.ylabel('Value')
  plt.yticks([0,1], ["Completely\nWrong", "Completely\nRight"])

  plt.sca(axs[2])
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

  plt.sca(axs[0])
  color = plt.plot(lrt, fpr[idx])[0].get_color()
  plt.plot(lrt, tpr[idx], color=color, dashes=[1,1])
  plt.legend(["FPR", "TPR"])
  plt.xlabel('Likelihood Ratio')
  plt.xscale("log")
