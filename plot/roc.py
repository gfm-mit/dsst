import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
import numpy as np

def get_roc(predicted, labels):
  points = pd.DataFrame(dict(predicted=predicted, labels=labels)).sort_values(by="predicted")
  points["tpr"] = 1 - (1+points.labels.cumsum() - points.labels) / (2+points.labels.sum())
  negs = 1 - points.labels
  points["fpr"] = 1 - (1+negs.cumsum() - negs) / (2+negs.sum())
  points["tpr_literal"] = 1 - (points.labels.cumsum() - points.labels) / (points.labels.sum())
  negs = 1 - points.labels
  points["fpr_literal"] = 1 - (negs.cumsum() - negs) / (negs.sum())
  return points

def get_roc_convex_hull(_, fpr, tpr):
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
  return pd.DataFrame(dict(fpr=fpr, tpr=tpr))

def interpolate(f1, f2, t1, t2):
  N = 100
  return np.stack([
    np.linspace(f1, f2, N), np.linspace(t1, t2, N)
  ]).T

def get_roc_interpolated_convex_hull(fpr, tpr):
  fn = iter(fpr)
  next(fn)
  tn = iter(tpr)
  next(tn)
  points = np.concatenate([
    interpolate(f1, f2, t1, t2)
    for f1, f2, t1, t2 in zip(fpr, fn, tpr, tn)
  ]).T
  fpr, tpr = points
  return pd.DataFrame(dict(fpr=fpr, tpr=tpr))

def get_slopes(fpr, tpr):
  slopes = np.diff(tpr) / np.diff(fpr)
  slopes[np.diff(fpr) == 0] = np.inf
  slopes[np.diff(tpr) == 0] = 0
  lr = np.geomspace(1e-2, 1e2, 100)
  idx = np.searchsorted(slopes, lr)
  counts = np.diff(tpr) + slopes * np.diff(fpr)
  counts /= 1 + slopes
  density = gaussian_kde(np.log(slopes)[1:-1], weights=counts[1:-1]).evaluate(np.log(lr))
  return density, lr, idx