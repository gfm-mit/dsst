import pandas as pd
import matplotlib.pyplot as plt
from plot.probit import *
from plot.log_one_minus import *
from plot.roc import *
from plot.plot_library import *
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, gaussian_kde

def get_3_axes():
  fig, axs = plt.subplots(1, 3, figsize=(8, 4), width_ratios=[1,2,1])
  return axs

def plot_3_types(predicted, labels, axs):
  register_scale(ProbitScale)
  register_scale(LogOneMinusXScale)
  roc = get_roc(predicted, labels)
  convex = get_roc_convex_hull(roc.shape[0], roc.fpr_literal.values, roc.tpr_literal.values)
  interpolated_smooth = get_roc_interpolated_convex_hull(convex.fpr, convex.tpr)
  eta_density, eta, idx = get_slopes(convex.fpr, convex.tpr)

  plt.sca(axs[1])
  color = plot_roc_swets(roc, convex, interpolated_smooth, axs, color=None)
  plt.sca(axs[0])
  plot_eta_percent(eta_density, eta, idx, roc, convex, interpolated_smooth, axs, color, labels.sum() / labels.size)
  plt.sca(axs[2])
  plot_precision_at_k(eta, idx, roc, convex, interpolated_smooth, axs, color)
  aucroc = roc_auc_score(labels, predicted)
  print("aucroc", aucroc)