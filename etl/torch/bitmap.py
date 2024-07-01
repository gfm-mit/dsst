import pathlib
from pathlib import Path
from hashlib import sha1

import numpy as np
import pandas as pd
import scipy
import torch


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, test_split=False, device="cpu", task=None):
        assert task in "classify classify_patient".split()
        self.test_split = test_split
        uniq_splits = np.unique(metadata.index)
        assert uniq_splits.shape[0] == 1, uniq_splits
        self.md = metadata.copy()
        features = []
        labels = []
        groups = []
        for _, (pkey, coarse) in self.md.iterrows():
          npz = Path('/Users/abe/Desktop/BITMAP/') / f"{pkey}.npz"
          data = scipy.sparse.load_npz(npz)
          N = np.sqrt(data.shape[1])
          assert N == int(N)
          N = int(N)
          features += [
             data[i].reshape([N, N])
             for i in range(data.shape[0])
          ]
          labels += [coarse] * data.shape[0]
          groups += [pkey] * data.shape[0]
        assert len(features)
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.groups = np.array(groups)
        self.device = device

    # turns out I accidentally transposed the indices in dataset
    # and it's too late to change the whole framework
    def __getitems__(self, indices):
      x, y, *g = zip(*[self.__getitem__(i) for i in indices])
      x = torch.cat(x, axis=0).unsqueeze(1)
      y = torch.cat(y, axis=0).unsqueeze(1)
      if self.test_split:
        g = g[0]
        return x, y, g
      else:
        return x, y

    def __getitem__(self, index):
        coarse = self.labels[[index]]
        nda = self.features[index].toarray()
        x = torch.Tensor(nda[np.newaxis, :, :]).to(self.device)
        y = torch.LongTensor(coarse).to(self.device)
        if self.test_split:
          g = self.groups[index]
          return x, y, g
        else:
           return x, y

    def __len__(self):
        return self.labels.shape[0]

def get_loaders(device=None, task=None):
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
  train_data = SeqDataset(labels.loc["train"], device=device, task=task)
  val_data = SeqDataset(labels.loc["val1"], device=device, task=task)
  calibration_data = SeqDataset(labels.loc["val1"], device=device, test_split=True, task=task)
  test_data = SeqDataset(labels.loc["val2"], device=device, test_split=True, task=task)
  # pin_memory is maybe a 30% speedup
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=lambda x: x)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True, collate_fn=lambda x: x)
  calibration_loader = torch.utils.data.DataLoader(calibration_data, batch_size=64, shuffle=True, collate_fn=lambda x: x)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, collate_fn=lambda x: x)
  return train_loader, val_loader, calibration_loader, test_loader