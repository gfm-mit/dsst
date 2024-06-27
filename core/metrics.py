import numpy as np
import pandas as pd
import sklearn
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
    return float(roc_auc_score(targets, logits[:, 1]))

def early_stop(history, task):
  if task == "next_token":
    return np.min(history)
  else:
    return np.max(history)

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
  logits, targets, groups = core.batch_eval.binary_classifier_with_groups(model, test_loader)
  if combine_fn is not None:
    logits, targets = combine_fn(logits, targets, groups)
  return logits, targets

def linear_combiner(logits, targets, groups, calibration_loader=None):
  df = pd.DataFrame(dict(logits=logits, targets=targets, groups=groups))
  df = df.groupby("groups").mean()
  logits, targets = df.logits, df.targets
  return logits, targets