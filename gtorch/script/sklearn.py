import pandas as pd
import numpy as np
import sys
import torch
from hashlib import sha1
import matplotlib.pyplot as plt
import argparse

from plot.palette import get_3_axes, plot_3_types, draw_3_legends
from sklearn.linear_model import LogisticRegression
import gtorch.datasets.synthetic
import gtorch.datasets.linear_patient
import gtorch.datasets.linear_box
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params
import util.excepthook


def main(train_loader, val_loader, test_loader, axs=None, device='cpu', random_state=None, solver=None):
  x, y = [z[0] for z in zip(*train_loader)]
  x = x[:, :, 0]
  y = y[:, 0]

  hash = int(sha1(bytes(str(y), 'utf8')).hexdigest(), 16) & ((1<<16) - 1)
  print(f"{hash:0b}", y[:4])
  model = LogisticRegression(random_state=random_state, solver=solver, max_iter=1 << 9)
  model.fit(x, y)
  print("bias", model.intercept_, "coef", model.coef_)

  x, y, g = [z[0] for z in zip(*test_loader)]
  x = x[:, :, 0]
  y = y[:, 0]
  logits = model.predict_log_proba(x)[:, 1]
  targets = y.numpy()
  df = pd.DataFrame(dict(logits=logits, targets=targets, groups=g))
  df = df.groupby("groups").mean()
  logits, targets = df.logits, df.targets

  axs = get_3_axes() if axs is None else axs
  line1 = plot_3_types(logits, targets, axs)
  return axs, line1
  draw_3_legends(axs, [line1])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Use SKLearn on the pytorch data loader')
  parser.add_argument('--solver', default='lbfgs', help='SKLearn Logistic Regresion Solver')
  args = parser.parse_args()
  axs = None
  lines = []
  sys.excepthook = util.excepthook.custom_excepthook
  train_loader, val_loader, test_loader = gtorch.datasets.linear_patient.get_loaders()
  axs, line1 = main(train_loader, val_loader, test_loader, axs=axs, device='cpu', random_state=42, solver=args.solver)
  lines += [line1]
  #train_loader, val_loader, test_loader = gtorch.datasets.linear.get_loaders()
  #axs, line2 = main(train_loader, val_loader, test_loader, axs=axs)
  draw_3_legends(axs, lines)