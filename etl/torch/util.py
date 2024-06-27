import numpy as np
import pandas as pd


def linear_combiner(logits, targets, groups, calibration_loader=None):
  df = pd.DataFrame(dict(logits=logits, targets=targets, groups=groups))
  df = df.groupby("groups").mean()
  logits, targets = df.logits, df.targets
  return logits, targets

def logistic_combiner(logits, targets, groups, calibration_loader=None):
  df = pd.DataFrame(dict(logits=logits, targets=targets, groups=groups))
  df = df.groupby("groups").mean()
  logits, targets = df.logits, df.targets
  return logits, targets