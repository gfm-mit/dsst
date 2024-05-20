import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pathlib
import scipy.special
from hashlib import sha1

LINEAR_AGG_COV = np.array([
 [877.460, -0.543, 0.228, -0.116, 1.000, -0.845, -455.043, -2.534, 0.000, 0.318, -1.197, -2.950],
 [-0.543, 0.371, 0.295, 0.068, 0.016, 0.243, -2.068, 0.059, 0.000, -0.074, -0.036, 0.310],
 [0.228, 0.295, 0.420, 0.069, 0.045, 0.323, -2.145, 0.061, 0.000, -0.072, -0.030, 0.151],
 [-0.116, 0.068, 0.069, 0.015, 0.007, 0.060, -0.532, 0.003, 0.000, -0.014, -0.006, 0.036],
 [1.000, 0.016, 0.045, 0.007, 0.070, 0.051, -0.055, 0.041, 0.000, -0.005, -0.005, 0.062],
 [-0.845, 0.243, 0.323, 0.060, 0.051, 0.340, -2.751, 0.024, 0.000, -0.063, -0.023, 0.010],
 [-455.043, -2.068, -2.145, -0.532, -0.055, -2.751, 785.121, 0.744, 0.000, 0.485, 1.424, 2.788],
 [-2.534, 0.059, 0.061, 0.003, 0.041, 0.024, 0.744, 0.838, 0.000, -0.023, -0.067, 0.993],
 [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
 [0.318, -0.074, -0.072, -0.014, -0.005, -0.063, 0.485, -0.023, 0.000, 0.018, 0.009, -0.063],
 [-1.197, -0.036, -0.030, -0.006, -0.005, -0.023, 1.424, -0.067, 0.000, 0.009, 0.057, -0.170],
 [-2.950, 0.310, 0.151, 0.036, 0.062, 0.010, 2.788, 0.993, 0.000, -0.063, -0.170, 3.676]
 ])

def synth_cov(corr):
  inv_cov = np.eye(12)
  inv_cov -= corr * np.diag(np.ones(11), k=1)
  inv_cov -= corr * np.diag(np.ones(11), k=-1)
  cov = np.linalg.inv(inv_cov)
  return cov

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, test_split=False, corr=0):
      self.N = 200
      self.test_split = test_split
      cov = LINEAR_AGG_COV
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