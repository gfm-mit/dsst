import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import torch

from etl.parse_semantics import *
from etl.parse_dynamics import *

from plot.palette import *
import gtorch.datasets.linear_agg
import gtorch.datasets.linear
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params
import cProfile

class First2:
  def __init__(self, iterable):
    self.iterable = iterable

  def __iter__(self):
    for x, y, _ in self.iterable:
      yield x, y


def main(train_loader, val_loader, test_loader, axs=None, device='cpu'):
  x, y = [z[0] for z in zip(*train_loader)]
  x = x[:, :, 0]
  y = y[:, 0]

  print(np.hstack([x, y[:, np.newaxis]])[:2])
  model = LogisticRegression()
  model.fit(x, y)
  print("coef", model.coef_)

  x, y = [z[0] for z in zip(*train_loader)]
  x = x[:, :, 0]
  y = y[:, 0]
  logits = model.predict_log_proba(x)[:, 1]
  targets = y.numpy()

  axs = get_3_axes() if axs is None else axs
  line1 = plot_3_types(logits, targets, axs)
  return axs, line1
  draw_3_legends(axs, [line1])

if __name__ == "__main__":
  axs = None
  train_loader, val_loader, test_loader = gtorch.datasets.linear_agg.get_loaders()
  # TODO: evil!  val and test on train set
  axs, line1 = main(train_loader, val_loader, test_loader, axs=axs, device='cpu')
  train_loader, val_loader, test_loader = gtorch.datasets.linear.get_loaders()
  axs, line2 = main(train_loader, val_loader, test_loader, axs=axs)
  draw_3_legends(axs, [line1, line2])
  #cProfile.run('main()', 'output_file.prof')