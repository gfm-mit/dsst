import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.linear_model
import torch
from sklearn.metrics import roc_auc_score

import core.batch_eval

def evaluate(model, val_loader, task, offset=1):
  if task == "next_token":
    predicted, data = core.batch_eval.next_token(model, val_loader, offset=offset)
    if False:
      verbose_next_token_metrics(predicted, data)
    rmse = np.sqrt(float(np.mean((predicted - data)**2)))
    return rmse
  else:
    logits, targets = core.batch_eval.binary_classifier(model, val_loader)
    return float(roc_auc_score(targets, logits))

def early_stop(history, task):
  if task == "next_token":
    return np.min(history)
  else:
    return np.max(history)

def best_so_far(history, task):
  if not history:
    return True
  if task == "next_token":
    return np.argmin(history) == len(history) - 1
  else:
    return np.argmax(history) == len(history) - 1

def verbose_next_token_metrics(predicted, data):
    var = np.mean((data)**2, axis=(0, 1))
    mse = np.mean((predicted - data)**2, axis=(0, 1))
    r2 = np.mean(np.reshape(mse / var, [6, 2]), axis=1)
    with pd.option_context('display.float_format', '{:.1f}%'.format):
      print(pd.Series(100 * r2,
                      index="t v_mag2 a_mag2 dv_mag2 cw j_mag2".split(),
                      name="Verbose MSE components"
                      ))

def get_combined_roc(model, test_loader, combine_fn=None, calibration_loader=None):
  regression = get_regressor(model, test_loader, combine_fn, calibration_loader)
  logits, targets, groups = core.batch_eval.binary_classifier(model, test_loader)
  if combine_fn is not None:
    logits, targets = combine_fn(logits, targets, groups)
  if regression is not None:
    logits = regression.predict_proba(logits.values)[:, 1]
  return logits, targets

def get_regressor(model, test_loader, combine_fn, calibration_loader):
  if combine_fn != symbol_box_combiner:
    return None
  assert calibration_loader is not None
  logits, targets, groups = core.batch_eval.binary_classifier(model, test_loader)
  logits, targets = combine_fn(logits, targets, groups)
  regression = sklearn.linear_model.LogisticRegression()
  regression.fit(logits.values, targets)
  return regression

def linear_combiner(logits, targets, groups, calibration_loader=None):
  df = pd.DataFrame(groups, columns="symbol task box pkey".split()).drop(columns="symbol task box".split())
  df["logits"] = logits
  df["targets"] = targets
  df = df.groupby("pkey").mean()
  logits, targets = df.logits, df.targets
  return logits, targets

def symbol_box_combiner(logits, targets, groups, calibration_loader=None):
  grouped_logits = pd.DataFrame(groups, columns="symbol task box pkey".split()).drop(columns="box")
  grouped_logits["logits"] = logits
  grouped_logits["targets"] = targets
  task = grouped_logits.groupby("pkey targets task".split()).logits.mean().unstack()
  symbol = grouped_logits.groupby("pkey targets symbol".split()).logits.mean().unstack()
  grouped_logits = pd.concat([task, symbol], axis=1).reset_index().set_index("pkey")
  targets = grouped_logits.pop("targets")
  return grouped_logits, targets