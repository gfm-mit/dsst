import pathlib
from hashlib import sha1
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def get_trunc_minmax(trunc):
  def get_minmax(data):
    return np.concatenate([
      data.values[:trunc, 3:],
    ], axis=1)
  return get_minmax

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, test_split=False, device=None, task=None, trunc=128):
        self.device = device
        self.test_split = test_split
        uniq_splits = np.unique(metadata.index)
        assert uniq_splits.shape[0] == 1, uniq_splits
        self.md = metadata.copy()
        features = []
        labels = []
        groups = []
        for _, (pkey, coarse) in self.md.iterrows():
          csv = Path('/Users/abe/Desktop/NP/') / f"{pkey}.npy"
          data = np.load(csv)
          data = pd.DataFrame(data, columns="symbol task box t v_mag2 a_mag2 dv_mag2 cw j_mag2".split())
          data = data.groupby("symbol task box".split()).apply(get_trunc_minmax(trunc))
          features += [data.values]

          idx = data.index.to_frame()
          if task == "classify_section":
            labels += idx.task.values.tolist()
          elif task == "next_token":
            labels += [0] * data.shape[0]
          else:
            labels += [coarse] * data.shape[0]
          idx["pkey"] = pkey
          groups += idx.values.tolist()
        assert len(features)
        self.features = np.concatenate(features)
        self.labels = np.array(labels)
        self.groups = np.array(groups)
        self.cache = {}

    def __getitems__(self, indices):
      if self.device is None or self.device == "cpu":
        return self.cache_miss(indices)
      if str(indices) not in self.cache:
        self.cache[str(indices)] = self.cache_miss(indices)
      return self.cache[str(indices)]

    def cache_miss(self, indices, device=None):
        coarse = self.labels[indices, np.newaxis]
        nda = self.features[indices]

        max_len = max([x.shape[0] for x in nda])
        padded = np.zeros([len(indices), max_len, 6])
        for i in range(len(indices)):
          jagged = nda[i]
          padded[i, :jagged.shape[0]] = jagged
        np.nan_to_num(padded, copy=False)

        x = torch.Tensor(padded)
        y = torch.LongTensor(coarse)
        if self.test_split:
          g = self.groups[indices]
          return x, y, g
        else:
          if device is not None:
            return x.to(device), y.to(device)
          return x, y

    def __len__(self):
        return self.labels.shape[0]

def get_loaders(device=None, task=None, batch_size=1000, shuffle=True, trunc=128):
  # note full batch of data is 4833
  labels = pd.read_csv(pathlib.Path("/Users/abe/Desktop/meta.csv")).set_index("AnonymizedID")
  #assert(features.index.difference(labels.index).empty), (features.index, labels.index)
  labels = labels.Diagnosis[labels.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])] == "Dementia-AD senile onset"
  labels = labels.reset_index()
  split = labels.index.astype(str).map(lambda x: int(sha1(bytes(x, 'utf8')).hexdigest(), 16) % 6)
  labels["split"] = np.select(
    [split == 0, split == 1, split == 2],
    "test val2 val1".split(),
    default="train")
  labels = labels.set_index("split")
  train_data = SeqDataset(labels.loc["train"], device=device, task=task, trunc=trunc)
  val_data = SeqDataset(labels.loc["val1"], device=device, task=task, trunc=trunc)
  calibration_data = SeqDataset(labels.loc["val1"], device=device, test_split=True, task=task, trunc=trunc)
  test_data = SeqDataset(labels.loc["val2"], device=device, test_split=True, task=task, trunc=trunc)
  # pin_memory is maybe a 30% speedup
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
  calibration_loader = torch.utils.data.DataLoader(calibration_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
  return train_loader, val_loader, calibration_loader, test_loader