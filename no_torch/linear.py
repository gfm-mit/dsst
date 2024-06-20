import pathlib

from matplotlib import pyplot as plt
import pandas as pd

from no_torch.linear_aggregate import get_predictions


def read_dynamics(kind="graph"):
  dynamics_path = pathlib.Path("/Users/abe/Desktop/DYNAMICS/")
  files = list(dynamics_path.glob("*.csv"))
  summary = []
  for csv in files:
    subset = csv.stem
    df = pd.read_csv(csv).drop(columns="x y t row_number".split())
    if kind == "graph":
      df = df.groupby("symbol task box".split()).apply(lambda g: g.mean(skipna=True)).reset_index(drop=True)
      df["box"] = df.groupby('symbol task'.split()).cumcount() + 1
      df = df.query("task in [1, 3] and box < 8")
      df = df.set_index("task symbol box".split())
      df = df.loc[3] - df.loc[1]
      df = df.groupby("symbol").mean().median().rename(subset)
    else:
      df = df.drop(columns="symbol task box".split()).mean(skipna=True).rename(subset)
    summary += [df]
  summary = pd.concat(summary, axis=1).T
  summary.to_csv(pathlib.Path("/Users/abe/Desktop/features.csv"))

def make_plots():
  y_hat, y = get_predictions(pathlib.Path("/Users/abe/Desktop/features.csv"), weight_ratio=1e3)
  plt.plot(y_hat, y)
  y_hat, y = get_predictions(pathlib.Path("/Users/abe/Desktop/features.csv"), weight_ratio=1e-1)
  plt.plot(y_hat, y)
  plt.show()

if __name__ == "__main__":
  #read_dynamics(graph=True)
  make_plots()