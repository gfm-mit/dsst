import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.scale import register_scale
from plot.probit import ProbitScale
from plot.log_one_minus import LogOneMinusXScale
from plot.roc import get_roc, get_roc_convex_hull, get_roc_interpolated_convex_hull, get_slopes
from plot.plot_library import plot_eta_percent, plot_precision_at_k, plot_roc_swets, plot_precision_recall, plot_supply_demand
from sklearn.metrics import roc_auc_score, average_precision_score
import pathlib
from sklearn.linear_model import LogisticRegression

def get_3_axes():
  fig, axs = plt.subplots(2, 2, figsize=(8, 8))#(8, 4), width_ratios=[1,2,1])
  axs = axs.flatten()
  return axs

def plot_3_types(predicted, labels, axs):
  register_scale(ProbitScale)
  register_scale(LogOneMinusXScale)
  #print("predicted, labels", predicted, labels)
  roc = get_roc(predicted, labels)
  convex = get_roc_convex_hull(roc.fpr_literal.values, roc.tpr_literal.values)
  interpolated_smooth = get_roc_interpolated_convex_hull(convex.fpr, convex.tpr)
  #print("convex.shape", convex.shape)
  eta_density, eta, idx = get_slopes(convex.fpr, convex.tpr)

  plt.sca(axs[1])
  artist, color = plot_roc_swets(roc, convex, interpolated_smooth, axs, color=None)
  plt.sca(axs[0])
  color, artist_eta, avg_eta = plot_eta_percent(eta_density, eta, idx, roc, convex, interpolated_smooth, axs, color, labels.sum() / labels.size)
  plt.sca(axs[2])
  color, artist_pr, avg_pr = plot_precision_at_k(eta, idx, roc, convex, interpolated_smooth, axs, color)
  plt.sca(axs[3])
  plot_supply_demand(eta, idx, roc, convex, interpolated_smooth, axs, color)

  aucroc = 100 * roc_auc_score(labels, predicted)
  return dict(
    roc_label=f"AUC {aucroc:.1f}%",
    roc_artist=artist,
    pr_label=f"Avg P {avg_pr:.1f}%",
    pr_artist=artist_pr,
    eta_label=f"Avg U {avg_eta:.1f}%",
    eta_artist=artist_eta,
  )

def draw_3_legends(axs, dicts):
  plt.sca(axs[0])
  plt.legend([x["eta_artist"] for x in dicts], [x["eta_label"] for x in dicts], loc="upper right")
  plt.sca(axs[1])
  plt.legend([x["roc_artist"] for x in dicts], [x["roc_label"] for x in dicts], loc="lower right")
  plt.sca(axs[2])
  plt.legend([x["pr_artist"] for x in dicts], [x["pr_label"] for x in dicts], loc="lower left")
  plt.tight_layout()
  plt.show()