# flake8: noqa
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def get_predictions(path, weight_ratio=1):
  features = pd.read_csv(path).set_index("Unnamed: 0")
  labels = pd.read_csv(pathlib.Path("/Users/abe/Desktop/meta.csv")).set_index("AnonymizedID")
  assert(features.index.difference(labels.index).empty), (features.index, labels.index)
  labels = labels.Diagnosis[labels.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])] == "Dementia-AD senile onset"
  train = labels[labels.index.astype(str).map(hash).astype(np.uint64) % 5 > 1]
  validation = labels[labels.index.astype(str).map(hash).astype(np.uint64) % 5 == 1]
  V = train

  model = LogisticRegression()
  model.fit(features.reindex(train.index).values, train, sample_weight=train.values + (1-train.values) * weight_ratio)
  #print(pd.Series(model.coef_[0], index=features.columns))
  y_hat = model.predict_proba(features.reindex(V.index).values)[:, 1]
  return y_hat, V.values