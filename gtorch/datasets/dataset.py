import numpy as np
import pandas as pd
import torch
from pathlib import Path
import einops
import pathlib

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, trunc=512):
        uniq_splits = np.unique(metadata.index)
        assert uniq_splits.shape[0] == 1, uniq_splits
        self.md = metadata.copy()
        self.md.coarse = self.md.coarse.cat.codes
        self.trunc = trunc
        rows = []
        for _, (pkey, coarse) in self.md.iterrows():
          for npy in (Path('TS/') / pkey).glob('**/*.npy'):
            rows += [[*npy.parts, npy, coarse]]
        assert len(rows)
        self.files = pd.DataFrame(rows)
        self.files.columns = "folder pkey symbol task iteration path coarse".split()
        self.files = self.files.drop(columns="folder iteration".split())
        self.files = self.files.set_index("symbol pkey task".split())

    def __getitem__(self, index):
        path, coarse = self.files.iloc[index]
        nda = np.load(str(path))
        x = torch.Tensor(nda[:self.trunc, :])
        y = torch.LongTensor([coarse])
        return x, y

    def __len__(self):
        return self.files.shape[0]

def collate_fn_padd(batch):
    features, targets = zip(*batch)

    max_l = np.max([nda.shape[0] for nda in features])

    # TODO: silly to go to numpy and back to tensors
    def pad_to(seq, l):
      dl = l - seq.shape[0]
      return np.pad(seq, [(0, dl), (0, 0)])
    features = np.stack([
        pad_to(x, max_l)
        for x in features
    ])
    features = np.nan_to_num(features, 0)
    features = einops.rearrange(features, 'b l c -> b c l')
    return torch.Tensor(features), torch.Tensor(targets).long()

def get_loaders():
  labels = pd.read_csv(pathlib.Path("/Users/abe/Desktop/meta.csv")).set_index("AnonymizedID")
  #assert(features.index.difference(labels.index).empty), (features.index, labels.index)
  labels = labels.Diagnosis[labels.Diagnosis.isin(["Healthy Control", "Dementia-AD senile onset"])] == "Dementia-AD senile onset"
  train = labels[labels.index.astype(str).map(hash).astype(np.uint64) % 5 > 1]
  validation = labels[labels.index.astype(str).map(hash).astype(np.uint64) % 5 == 1]
  print(train.head())
  #train_data = SeqDataset(metadata.loc["train"])
  #val_data = SeqDataset(metadata.loc["val"])
  #test_data = SeqDataset(metadata.loc["test"])
  #train_loader = torch.utils.data.DataLoader(
  #    train_data, batch_size=64, shuffle=True, collate_fn=collate_fn_padd)
  #val_loader = torch.utils.data.DataLoader(
  #    val_data, batch_size=1000, shuffle=True, collate_fn=collate_fn_padd)
  #test_loader = torch.utils.data.DataLoader(
  #    test_data, batch_size=1000, shuffle=True, collate_fn=collate_fn_padd)

  #z = next(iter(train_loader))
  #z[0].shape, z[1].shape, z[0].sum(), z[1].sum()

  ##q = ModelWrapper(512)
  #q.reset()
  #train(0, q, FakeOptimizer(q.parameters()), 'cuda')
  ##tgt, pred, _ = test(q)
  #qq = pd.DataFrame(dict(target=tgt, predicted=pred))
  #qq = qq.target.groupby([qq.target, qq.predicted]).count().unstack()
  #qq.index = train_loader.dataset.translate(qq.index)
  #qq.columns = train_loader.dataset.translate(qq.columns)
  #np.diag(qq).sum() / qq.sum().sum()