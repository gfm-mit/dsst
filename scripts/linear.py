import random
from lxml import etree
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from etl.parse_semantics import *
from etl.parse_dynamics import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from plot.swets import *

def read_dynamics(graph=True):
  dynamics_path = pathlib.Path("/Users/abe/Desktop/DYNAMICS/")
  files = list(dynamics_path.glob("*.csv"))
  summary = []
  for csv in files:
    subset = csv.stem
    df = pd.read_csv(csv).drop(columns="x y t row_number".split())
    if graph:
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

def plot_features(path, axs, weight_ratio=1):
  features = pd.read_csv(path).set_index("Unnamed: 0")
  labels = pd.read_csv(pathlib.Path("/Users/abe/Desktop/meta.csv")).set_index("AnonymizedID")
  assert(features.index.difference(labels.index).empty), (features.index, labels.index)
  labels = labels.Diagnosis[labels.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])] == "Dementia-AD senile onset"
  train = labels[labels.index.astype(str).map(hash).astype(np.uint64) % 5 > 1]
  validation = labels[labels.index.astype(str).map(hash).astype(np.uint64) % 5 == 1]
  V = train
  #print(features.head().T)
  model = LogisticRegression()
  # Fit the model using features as the independent variable and labels as the dependent variable
  model.fit(features.reindex(train.index).values, train, sample_weight=train.values + (1-train.values) * weight_ratio)
  print(pd.Series(model.coef_[0], index=features.columns))
  y_hat = model.predict_proba(features.reindex(V.index).values)[:, 1]
  axs = axs.flatten()
  plot_9_types(y_hat, V.values, axs)

def make_plots():
  fig, axs = plt.subplots(3, 3, figsize=(12, 8))
  plot_features(pathlib.Path("/Users/abe/Desktop/features.csv"), axs, weight_ratio=1e3)
  plot_features(pathlib.Path("/Users/abe/Desktop/features.csv"), axs, weight_ratio=1e-1)
  #plot_features(pathlib.Path("/Users/abe/Desktop/features_graph.csv"), axs)
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  #read_dynamics(graph=True)
  make_plots()