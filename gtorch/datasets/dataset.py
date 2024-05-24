import pathlib
from hashlib import sha1
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def combiner(logits, targets, groups):
  df = pd.DataFrame(dict(logits=logits, targets=targets, groups=groups))
  df = df.groupby("groups").mean()
  logits, targets = df.logits, df.targets
  return logits, targets

def get_trunc_minmax(trunc):
  def get_minmax(data):
    return np.concatenate([
      data.values[:trunc, 3:],
      -data.values[:trunc, 3:],
    ], axis=1)
  return get_minmax

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, test_split=False):
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
          data = data.groupby("symbol task box".split()).apply(get_trunc_minmax(128))
          features += [data.values]
          labels += [coarse] * data.shape[0]
          groups += [pkey] * data.shape[0]
        assert len(features)
        self.features = np.concatenate(features)
        self.labels = np.array(labels)
        self.groups = np.array(groups)

    def __getitems__(self, indices):
        coarse = self.labels[indices, np.newaxis]
        nda = self.features[indices]

        max_len = max([x.shape[0] for x in nda])
        padded = np.zeros([len(indices), max_len, 12])
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
           return x, y

    def __len__(self):
        return self.labels.shape[0]

def get_loaders():
  labels = pd.read_csv(pathlib.Path("/Users/abe/Desktop/meta.csv")).set_index("AnonymizedID")
  #assert(features.index.difference(labels.index).empty), (features.index, labels.index)
  labels = labels.Diagnosis[labels.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])] == "Dementia-AD senile onset"
  labels = labels.reset_index()
  split = labels.index.astype(str).map(lambda x: int(sha1(bytes(x, 'utf8')).hexdigest(), 16) % 5)
  labels["split"] = np.where(split == 0, 'test', np.where(split == 1, 'validation', 'train'))
  labels = labels.set_index("split")
  train_data = SeqDataset(labels.loc["train"])
  val_data = SeqDataset(labels.loc["train"])
  test_data = SeqDataset(labels.loc["train"], test_split=True)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=False, collate_fn=lambda x: x)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000, shuffle=False, collate_fn=lambda x: x)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False, collate_fn=lambda x: x)
  return train_loader, val_loader, test_loader