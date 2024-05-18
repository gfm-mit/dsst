import pandas as pd
import sys

from plot.palette import get_3_axes, plot_3_types, draw_3_legends
from sklearn.linear_model import LogisticRegression
import gtorch.datasets.synthetic
import gtorch.datasets.linear_agg
import gtorch.datasets.linear
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params
import util.excepthook


def main(train_loader, val_loader, test_loader, axs=None, device='cpu', random_state=None):
  x, y = [z[0] for z in zip(*train_loader)]
  x = x[:, :, 0]
  y = y[:, 0]

  print(f"{hash(y):0b}", y[:4])
  model = LogisticRegression(random_state=random_state)
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
  axs = None
  lines = []
  sys.excepthook = util.excepthook.custom_excepthook
  train_loader, val_loader, test_loader = gtorch.datasets.linear_agg.get_loaders()
  for _ in range(3):
    axs, line1 = main(train_loader, val_loader, test_loader, axs=axs, device='cpu', random_state=42)
    lines += [line1]
  #train_loader, val_loader, test_loader = gtorch.datasets.linear.get_loaders()
  #axs, line2 = main(train_loader, val_loader, test_loader, axs=axs)
  draw_3_legends(axs, lines)