import pandas as pd
import matplotlib.pyplot as plt
from plot.probit import *
from plot.log_one_minus import *
from scipy.stats import norm
from scipy.spatial import ConvexHull

def get_roc(predicted, labels):
  points = pd.DataFrame(dict(predicted=predicted, labels=labels)).sort_values(by="predicted")
  points["tpr"] = 1 - (1+points.labels.cumsum() - points.labels) / (2+points.labels.sum())
  negs = 1 - points.labels
  points["fpr"] = 1 - (1+negs.cumsum() - negs) / (2+negs.sum())
  points["tpr_literal"] = 1 - (points.labels.cumsum() - points.labels) / (points.labels.sum())
  negs = 1 - points.labels
  points["fpr_literal"] = 1 - (negs.cumsum() - negs) / (negs.sum())
  return points

def plot_swets_roc(predicted, labels, axs):
  points = get_roc(predicted, labels)
  register_scale(ProbitScale)
  register_scale(LogOneMinusXScale)
  plt.sca(axs[4])
  color = plt.scatter(points.fpr, points.tpr, alpha=0.5).get_facecolor()[0]

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
  fpr, tpr = convex_hull_roc(points.shape[0], points.fpr_literal.values, points.tpr_literal.values)

  jank_fpr, jank_tpr = plot_convex_hull(fpr, tpr, color)
  equivalent_tpr = project(fpr, tpr, axs)

  plt.sca(axs[3])
  plt.plot(fpr, tpr, linewidth=3, alpha=0.2, color=color)
  plt.scatter(points.fpr_literal, points.tpr_literal, s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('FPR')
  plt.ylabel('TPR')

  plt.sca(axs[5])
  plt.plot(fpr, tpr, linewidth=3, alpha=0.2, color=color)
  plt.scatter(points.fpr_literal, points.tpr_literal, s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.xscale('log')

  plt.sca(axs[6])
  plt.plot(jank_tpr, jank_tpr / (jank_tpr + jank_fpr), linewidth=3, alpha=0.2, color=color)
  plt.scatter(points.tpr_literal, points.tpr_literal / (points.tpr_literal + points.fpr_literal), s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('Recall')
  plt.ylabel('Precision')

  plt.sca(axs[7])
  budget = jank_tpr * points.labels.sum() + jank_fpr * (1-points.labels).sum()
  nne = budget / jank_tpr / points.labels.sum()
  budget = budget / points.shape[0]
  plt.plot(budget, nne, linewidth=3, alpha=0.2, color=color)
  budget = points.tpr_literal * points.labels.sum() + points.fpr_literal * (1-points.labels).sum()
  nne = budget / points.tpr_literal / points.labels.sum()
  budget = budget / points.shape[0]
  plt.scatter(budget, nne, s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('Triage Budget')
  plt.xscale('log')
  plt.xlim([1e-2, 1])
  plt.ylabel('Number Needed to Evaluate')
  #plt.yscale('log1minusx')

  #plt.sca(axs[7])
  #pos_like = norm.pdf(norm.ppf(points.tpr_literal))
  #neg_like = norm.pdf(norm.ppf(points.fpr_literal))
  #snr = norm.ppf(points.tpr_literal) - norm.ppf(points.fpr_literal)
  #plt.scatter(neg_like / pos_like, snr, alpha=0.5)
  #plt.xscale('log')
  #plt.xlabel('Inv Binormal Like Ratio')
  #plt.ylabel('SNR\n(Sampled Points)')
  #axs[7].set_xlim([1e-2, 1e2])

  #plt.sca(axs[6])
  #pos_like = norm.pdf(norm.ppf(jank_tpr))
  #neg_like = norm.pdf(norm.ppf(jank_fpr))
  #snr = norm.ppf(jank_tpr) - norm.ppf(jank_fpr)
  #plt.scatter(neg_like / pos_like, snr, alpha=0.5)
  #plt.xscale('log')
  #plt.xlabel('Inv Binormal Like Ratio')
  #plt.ylabel('SNR\n(convex hull)')
  #axs[6].set_xlim(axs[7].get_xlim())
  #axs[6].set_ylim(axs[7].get_ylim())

  #plt.sca(axs[5])
  #snr = norm.ppf(points.tpr_literal) - norm.ppf(points.fpr_literal)
  #plt.plot(points.predicted, snr)

  #plt.sca(axs[6])
  #pos_like = norm.pdf(norm.ppf(points.tpr_literal))
  #neg_like = norm.pdf(norm.ppf(points.fpr_literal))
  #snr = norm.ppf(points.tpr_literal) - norm.ppf(points.fpr_literal)
  #plt.scatter(neg_like / pos_like, snr, alpha=0.5)
  #plt.xscale('log')
  #plt.xlabel('Inv. Binorm. Like. Ratio')
  #plt.ylabel('SNR\n(Sampled Points)')

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
  #plt.plot(lrt, y, linestyle="--", zorder=-10, alpha=0.2)
  #plt.plot(lrt, yy, color="lightgray", linestyle="--", zorder=-10)
  plt.xscale("log")
  plt.ylabel('Δ Value')
  plt.xlabel('Likelihood Ratio')

  #plt.sca(axs[6])
  #plt.plot(lrt, x - xx)
  ##plt.plot(lrt, x, linestyle="--", zorder=-10, alpha=0.2)
  ##plt.plot(lrt, xx, color="lightgray", linestyle="--", zorder=-10)
  #plt.xscale("log")
  #plt.xlabel('Likelihood Ratio')
  #plt.ylabel('Δ FPR')
  # Precision doesn't quite work, because it depends separately on the base rate.

  #plt.sca(axs[3])
  #plt.plot(lrt, (1e-4+tpr[idx]) / (1e-4+fpr[idx]), dashes=[2,2])
  ##print(tpr[idx] / (1e-4+fpr[idx]))
  #plt.plot([1, 1e2], [1, 1e2], color="lightgray", zorder=-10, alpha=0.5)
  #plt.xscale("log")
  #plt.yscale("log")
  #plt.xlabel('Likelihood Ratio')
  #plt.ylabel('Precision (Odds Ratio)')

  plt.sca(axs[8])
  plt.plot(lrt, tpr[idx] - fpr[idx])
  plt.xscale("log")
  plt.ylabel('Δ Equivalent TP\nwith zero FP')
  plt.xlabel('Likelihood Ratio\n(at Threshold, evaluation uses 1)')

  #plt.sca(axs[7])
  #i2 = np.searchsorted(slopes, 1)
  #plt.plot(lrt, np.maximum(0, tpr[i2] - lrt * fpr[i2]) - np.maximum(0, 1 - lrt))
  #plt.xscale("log")
  #plt.ylabel('Δ Equivalent TP\nwith zero FP')
  #plt.xlabel('Likelihood Ratio\n(for evaluation, threshold is 1)')

  plt.sca(axs[0])
  color = plt.plot(lrt, fpr[idx])[0].get_color()
  plt.plot(lrt, tpr[idx], color=color, dashes=[1,1])
  plt.legend(["FPR", "TPR"])
  plt.xlabel('Likelihood Ratio')
  plt.xscale("log")

def interpolate(f1, f2, t1, t2):
  N = 100
  return np.stack([
    np.linspace(f1, f2, N), np.linspace(t1, t2, N)
  ]).T

def convex_hull_roc(_, fpr, tpr):
  L = 0
  H = 1
  fpr = np.concatenate([[L, H, H], fpr])
  tpr = np.concatenate([[L, L, H], tpr])
  idx = ConvexHull(
      np.stack([fpr, tpr]).T,
      ).vertices
  idx = np.roll(idx, -1)[1:]
  fpr = fpr[idx]
  tpr = tpr[idx]
  return fpr, tpr

def plot_convex_hull(fpr, tpr, color):
  fn = iter(fpr)
  next(fn)
  tn = iter(tpr)
  next(tn)
  points = np.concatenate([
    interpolate(f1, f2, t1, t2)
    for f1, f2, t1, t2 in zip(fpr, fn, tpr, tn)
  ]).T
  fpr, tpr = points
  plt.plot(fpr, tpr, linewidth=3, alpha=0.2, color=color)
  return fpr, tpr