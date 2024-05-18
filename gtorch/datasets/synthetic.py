import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pathlib
import scipy.special
from hashlib import sha1

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, test_split=False, corr=0):
      self.N = 200
      self.test_split = test_split
      inv_cov = np.eye(12)
      inv_cov -= corr * np.diag(np.ones(11), k=1)
      inv_cov -= corr * np.diag(np.ones(11), k=-1)
      cov = np.linalg.inv(inv_cov)
      print(np.round(cov * 100).astype(int))
      self.files = pd.DataFrame(
        np.random.default_rng().multivariate_normal(np.zeros([12]), cov=cov, size=self.N)
      , columns="one two three four five six seven eight nine ten eleven twelve".split())
      logit = self.files.one + self.files.two + self.files.three + self.files.four + self.files.five + self.files.six
      logit -= self.files.seven + self.files.eight + self.files.nine + self.files.ten + self.files.eleven + self.files.twelve
      print("self.files.shape", self.files.shape)
      self.files["label"] = np.random.uniform(size=self.N) < scipy.special.expit(2 * logit)

    def __getitem__(self, index):
      row = self.files.iloc[index]
      x = torch.Tensor(row.iloc[:-1].astype(float).values[:, np.newaxis])
      y = torch.LongTensor([row.iloc[-1]])
      if self.test_split:
        return x, y, index
      else:
        return x, y

    def __len__(self):
        return self.N

def get_loaders():
  train_data = SeqDataset(corr=0.5)
  val_data = SeqDataset(corr=0.5)
  test_data = SeqDataset(corr=0.5, test_split=True)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=False)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=False)
  return train_loader, val_loader, test_loader