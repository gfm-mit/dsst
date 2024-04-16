import random
from lxml import etree
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from etl.parse_semantics import *
from etl.parse_dynamics import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def read_dynamics():
  dynamics_path = pathlib.Path("/Users/abe/Desktop/DYNAMICS/")
  files = list(dynamics_path.glob("*.csv"))
  summary = []
  for csv in files:
    subset = csv.stem
    df = pd.read_csv(csv).mean(skipna=True).rename(subset)
    summary += [df]
  summary = pd.concat(summary, axis=1).T
  summary.to_csv(pathlib.Path("/Users/abe/Desktop/features.csv"))

if __name__ == "__main__":
  features = pd.read_csv(pathlib.Path("/Users/abe/Desktop/features.csv")).set_index("Unnamed: 0").drop(columns="row_number box symbol task x y t".split())
  labels = pd.read_csv(pathlib.Path("/Users/abe/Desktop/meta.csv")).set_index("AnonymizedID")
  labels = labels.Diagnosis[labels.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])] == "Dementia-AD senile onset"
  labels = labels[labels.index.astype(str).map(hash).astype(np.uint64) % 5 > 0]
  print(features.head().T)
  model = LogisticRegression()
  # Fit the model using features as the independent variable and labels as the dependent variable
  model.fit(features.reindex(labels.index).values, labels)
  print(pd.Series(model.coef_[0], index=features.columns))
  y_hat = model.predict(features.reindex(labels.index).values)
  print(roc_auc_score(labels, y_hat))