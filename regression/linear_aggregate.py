import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.scale import register_scale
from plot.probit import ProbitScale
from plot.log_one_minus import LogOneMinusXScale
from plot.roc import get_roc, get_roc_convex_hull, get_roc_interpolated_convex_hull, get_slopes
from plot.plot_library import plot_eta_percent, plot_precision_at_k, plot_roc_swets
from sklearn.metrics import roc_auc_score, average_precision_score
import pathlib
from sklearn.linear_model import LogisticRegression

def get_3_axes():
  fig, axs = plt.subplots(1, 3, figsize=(8, 4), width_ratios=[1,2,1])
  return axs

def plot_3_types(predicted, labels, axs):
  register_scale(ProbitScale)
  register_scale(LogOneMinusXScale)
  print("predicted, labels", predicted, labels)
  roc = get_roc(predicted, labels)
  convex = get_roc_convex_hull(roc.shape[0], roc.fpr_literal.values, roc.tpr_literal.values)
  interpolated_smooth = get_roc_interpolated_convex_hull(convex.fpr, convex.tpr)
  print("convex.shape", convex.shape)
  eta_density, eta, idx = get_slopes(convex.fpr, convex.tpr)

  plt.sca(axs[1])
  artist, color = plot_roc_swets(roc, convex, interpolated_smooth, axs, color=None)
  plt.sca(axs[0])
  color, artist_eta, avg_eta = plot_eta_percent(eta_density, eta, idx, roc, convex, interpolated_smooth, axs, color, labels.sum() / labels.size)
  plt.sca(axs[2])
  color, artist_pr, avg_pr = plot_precision_at_k(eta, idx, roc, convex, interpolated_smooth, axs, color)

  aucroc = 100 * roc_auc_score(labels, predicted)
  return dict(
    roc_label=f"AUC {aucroc:.1f}%",
    roc_artist=artist,
    pr_label=f"Avg P {avg_pr:.1f}%",
    pr_artist=artist_pr,
    eta_label=f"Avg U {avg_eta:.1f}%",
    eta_artist=artist_eta,
  )

def get_predictions(path, weight_ratio=1):
  features = pd.read_csv(path).set_index("Unnamed: 0")
  labels = pd.read_csv(pathlib.Path("/Users/abe/Desktop/meta.csv")).set_index("AnonymizedID")
  assert(features.index.difference(labels.index).empty), (features.index, labels.index)
  labels = labels.Diagnosis[labels.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])] == "Dementia-AD senile onset"
  train = labels[labels.index.astype(str).map(hash).astype(np.uint64) % 5 > 1]
  validation = labels[labels.index.astype(str).map(hash).astype(np.uint64) % 5 == 1]
  V = train

  model = LogisticRegression()
  model.fit(features.reindex(train.index).values, train, sample_weight=train.values + (1-train.values) * weight_ratio)
  #print(pd.Series(model.coef_[0], index=features.columns))
  y_hat = model.predict_proba(features.reindex(V.index).values)[:, 1]
  return y_hat, V.values