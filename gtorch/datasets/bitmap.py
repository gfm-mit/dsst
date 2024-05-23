import numpy as np
import pandas as pd
import torch
from pathlib import Path
import scipy
import pathlib

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

    def __getitem__(self, index):
        coarse = self.labels[[index]]
        nda = self.features[index].toarray()
        x = torch.Tensor(nda[np.newaxis, :, :])
        y = torch.LongTensor(coarse)
        if self.test_split:
          g = self.groups[index]
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
  split = labels.index.astype(str).map(hash).astype(np.uint64) % 5
  labels["split"] = np.where(split == 0, 'test', np.where(split == 1, 'validation', 'train'))
  labels = labels.set_index("split")
  train_data = SeqDataset(labels.loc["train"])
  val_data = SeqDataset(labels.loc["train"])
  test_data = SeqDataset(labels.loc["train"], test_split=True)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
  return train_loader, val_loader, test_loader