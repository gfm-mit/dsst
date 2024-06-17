import numpy as np
import pandas as pd
import scipy
from scipy.spatial import ConvexHull
from sklearn import linear_model

def get_roc_convex(roc):
  roc = roc.sort_values(by='logits', ascending=False).reset_index(drop=True)
  roc['tpr_empirical'] = roc.targets.cumsum() / roc.targets.sum()
  roc['fpr_empirical'] = (1 - roc.targets).cumsum() / (1 - roc.targets).sum()
  # add (0,0) to the high end
  roc.loc[-1] = [np.inf, np.nan, 0, 0]
  roc['y_hat'] = scipy.special.expit(roc.logits)
  roc = roc.sort_index()

  # get the convex hull
  idx = ConvexHull(roc["fpr_empirical tpr_empirical".split()]).vertices
  # make sure it starts with the smallest number
  idx = np.roll(idx[::-1], 1) - 1
  # take differences of the bins it establishes
  deltas = roc.loc[idx].diff()
  # set that binned prediction for every point in the bin
  for idx1, idx2 in zip(idx[:-1], idx[1:]):
    row = deltas.loc[idx2]
    y_hat = row.tpr_empirical / (row.tpr_empirical + row.fpr_empirical)
    roc.loc[idx1:idx2, 'y_convex'] = y_hat
  roc['tpr_convex'] = roc.y_convex.cumsum() / roc.y_convex.sum()
  roc['fpr_convex'] = (1 - roc.y_convex).cumsum() / (1 - roc.y_convex).sum()
  return roc

def get_costs(roc):
  skew = roc.y_hat
  roc["cost_empirical"] = roc.fpr_empirical * skew + (1 - skew) * (1 - roc.tpr_empirical)
  skew = roc.y_logistic
  roc["cost_logistic"] = roc.fpr_logistic * skew + (1 - skew) * (1 - roc.tpr_logistic)
  skew = roc.y_convex
  roc["cost_convex"] = roc.fpr_convex * skew + (1 - skew) * (1 - roc.tpr_convex)
  skew = roc.y_hat
  roc["cost_hat"] = roc.fpr_hat * skew + (1 - skew) * (1 - roc.tpr_hat)
  return roc

def get_roc_table(roc):
  roc = get_roc_convex(roc)
  roc['fpr_hat'] = (1 - roc.y_hat).cumsum() / (1 - roc.y_hat).sum()
  roc['tpr_hat'] = roc.y_hat.cumsum() / roc.y_hat.sum()

  roc["y_logistic"] = 1
  valid = roc.iloc[1:]

  rescaled = valid.logits - valid.logits.mean()
  rescaled = rescaled / rescaled.std()
  X = np.stack([
    rescaled,
    #3e-2 * rescaled ** 2,  # this is a hack to keep monotonicity
  ], axis=1)
  roc.loc[0:, "y_logistic"] = linear_model.LogisticRegression().fit(X, valid.targets.values).predict_proba(X)[:, 1]
  roc['tpr_logistic'] = roc.y_logistic.cumsum() / roc.y_logistic.sum()
  roc['fpr_logistic'] = (1 - roc.y_logistic).cumsum() / (1 - roc.y_logistic).sum()

  roc = get_costs(roc)
  return roc

def get_full_roc_table(logits, targets):
  roc = pd.DataFrame(dict(logits=logits, targets=targets))
  roc = get_roc_table(roc)
  return roc