import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pathlib
from hashlib import sha1

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, test_split=False):
        self.test_split = test_split
        uniq_splits = np.unique(metadata.index)
        assert uniq_splits.shape[0] == 1, uniq_splits
        self.md = metadata.copy()
        rows = []
        for _, (pkey, coarse) in self.md.iterrows():
          csv = Path('/Users/abe/Desktop/NP/') / f"{pkey}.npy"
          data = np.nanmean(np.load(csv), axis=0)
          data = pd.Series(data, index="symbol task box t v_mag2 a_mag2 dv_mag2 cw j_mag2".split())
          data["label"] = coarse
          data = data.drop("symbol task box".split()).rename(pkey)
          rows += [data]
        assert len(rows)
        self.files = pd.DataFrame(rows)
        for c in "t v_mag2 a_mag2 dv_mag2 cw j_mag2".split():
          self.files[c] = self.files[c] - self.files[c].mean()

    def __getitem__(self, index):
        coarse = self.files.iloc[index, -1]
        nda = self.files.iloc[index, :-1].astype(float).values[:, np.newaxis]
        x = torch.Tensor(nda[:, :])
        y = torch.LongTensor([coarse])
        if self.test_split:
          return x, y, index
        else:
          return x, y

    def __len__(self):
        return self.files.shape[0]

def get_loaders():
  labels = pd.read_csv(pathlib.Path("/Users/abe/Desktop/meta.csv")).set_index("AnonymizedID").sort_index()
  #assert(features.index.difference(labels.index).empty), (features.index, labels.index)
  labels = labels.Diagnosis[labels.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])] == "Dementia-AD senile onset"
  labels = labels.reset_index()
  split = labels.index.astype(str).map(lambda x: int(sha1(bytes(x, 'utf8')).hexdigest(),  16) % 5)
  labels["split"] = np.where(split == 0, 'test', np.where(split == 1, 'validation', 'train'))
  labels = labels.set_index("split")
  train_data = SeqDataset(labels.loc["train"])
  val_data = SeqDataset(labels.loc["train"])
  #test_data = SeqDataset(labels.loc["test"], test=True)
  test_data = SeqDataset(labels.loc["train"], test_split=True)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=False)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)
  return train_loader, val_loader, test_loader