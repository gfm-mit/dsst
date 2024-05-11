import numpy as np
import pandas as pd
import torch
import shutil
from pathlib import Path
import re
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import einops
import pathlib

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, trunc=512):
        self.trunc = trunc
        uniq_splits = np.unique(metadata.index)
        assert uniq_splits.shape[0] == 1, uniq_splits
        self.md = metadata.copy()
        rows = []
        for _, (pkey, coarse) in self.md.iterrows():
          csv = Path('/Users/abe/Desktop/BITMAP/') / f"{pkey}.csv"
          npz = Path('/Users/abe/Desktop/BITMAP/') / f"{pkey}.npz"
          meta = pd.read_csv(csv)
          data = scipy.sparse.load_npz(npz)
          meta["bitmap"] = [
             data[i].reshape([64, 64])
             for i in range(data.shape[0])
          ]
          meta["pkey"] = pkey
          meta["label"] = coarse
          meta = meta.set_index("pkey")
          rows += [meta]
        assert len(rows)
        self.files = pd.concat(rows, axis=0)

    def __getitem__(self, index):
        coarse = self.files.iloc[index, -1]
        nda = self.files.iloc[index, -2].toarray()
        x = torch.Tensor(nda[np.newaxis, :, :])
        y = torch.LongTensor([coarse])
        return x, y

    def __len__(self):
        return self.files.shape[0]

def get_loaders():
  labels = pd.read_csv(pathlib.Path("/Users/abe/Desktop/meta.csv")).set_index("AnonymizedID")
  #assert(features.index.difference(labels.index).empty), (features.index, labels.index)
  labels = labels.Diagnosis[labels.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])] == "Dementia-AD senile onset"
  labels = labels.reset_index()
  split = labels.index.astype(str).map(hash).astype(np.uint64) % 5
  labels["split"] = np.where(split == 0, 'test', np.where(split == 1, 'validation', 'train'))
  labels = labels.set_index("split")
  train_data = SeqDataset(labels.loc["train"])
  #val_data = SeqDataset(metadata.loc["val"])
  #test_data = SeqDataset(metadata.loc["test"])
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
  #val_loader = torch.utils.data.DataLoader(
  #    val_data, batch_size=1000, shuffle=True, collate_fn=collate_fn_padd)
  #test_loader = torch.utils.data.DataLoader(
  #    test_data, batch_size=1000, shuffle=True, collate_fn=collate_fn_padd)
  return train_loader