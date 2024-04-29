import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import torch

from etl.parse_semantics import *
from etl.parse_dynamics import *

from regression.linear_aggregate import *
import gtorch.datasets.linear
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params

#def read_dynamics(kind="graph"):
#  dynamics_path = pathlib.Path("/Users/abe/Desktop/DYNAMICS/")
#  files = list(dynamics_path.glob("*.csv"))
#  summary = []
#  for csv in files:
#    subset = csv.stem
#    df = pd.read_csv(csv).drop(columns="x y t row_number".split())
#    if kind == "graph":
#      df = df.groupby("symbol task box".split()).apply(lambda g: g.mean(skipna=True)).reset_index(drop=True)
#      df["box"] = df.groupby('symbol task'.split()).cumcount() + 1
#      df = df.query("task in [1, 3] and box < 8")
#      df = df.set_index("task symbol box".split())
#      df = df.loc[3] - df.loc[1]
#      df = df.groupby("symbol").mean().median().rename(subset)
#    else:
#      df = df.drop(columns="symbol task box".split()).mean(skipna=True).rename(subset)
#    summary += [df]
#  summary = pd.concat(summary, axis=1).T
#  summary.to_csv(pathlib.Path("/Users/abe/Desktop/features.csv"))
#
#def make_plots():
#  axs = get_3_axes()
#  axs = axs.flatten()
#  y_hat, y = get_predictions(pathlib.Path("/Users/abe/Desktop/features.csv"), weight_ratio=1e3)
#  line1 = plot_3_types(y_hat, y, axs)
#  y_hat, y = get_predictions(pathlib.Path("/Users/abe/Desktop/features.csv"), weight_ratio=1e-1)
#  line2 = plot_3_types(y_hat, y, axs)
#  plt.tight_layout()
#  plt.sca(axs[0])
#  plt.legend([x["eta_artist"] for x in [line1, line2]], [x["eta_label"] for x in [line1, line2]], loc="upper right")
#  plt.sca(axs[1])
#  plt.legend([x["roc_artist"] for x in [line1, line2]], [x["roc_label"] for x in [line1, line2]], loc="lower right")
#  plt.sca(axs[2])
#  plt.legend([x["pr_artist"] for x in [line1, line2]], [x["pr_label"] for x in [line1, line2]], loc="lower left")
#  plt.show()

if __name__ == "__main__":
  train_loader = gtorch.datasets.linear.get_loaders()
  val_loader = train_loader #TODO: evil!!
  model, base_params = gtorch.models.linear.get_model(hidden_width=2, device='cpu', classes=1)
  #r = q(next(iter(train_loader))[0].to('cpu'))
  #gtorch.optimize.optimize.optimize(0, q, gtorch.optimize.optimize.FakeOptimizer(q), train_loader)
  #loss, accuracy = gtorch.optimize.optimize.metrics(q, val_loader=train_loader)
  retval, model = gtorch.hyper.params.many_hyperparams(base_params, model_factory_fn=gtorch.models.linear.get_model,
                                                       train_loader=train_loader, val_loader=val_loader)

  DEVICE = 'cpu'
  results = []
  model.eval()
  with torch.no_grad():
    for data, target in val_loader:
      print(target)
      output = model(data.to(DEVICE)).to('cpu')
      results += [(
          output.detach().numpy(),
          target.detach().to('cpu').numpy(),
      )]
  logits, targets = zip(*results)

  axs = get_3_axes()
  logits = logits[0][:, 0]
  targets = targets[0][:, 0]
  print(pd.DataFrame(dict(logits=logits, targets=targets)).head())
  line1 = plot_3_types(logits, targets, axs)
  plt.tight_layout()
  plt.show()