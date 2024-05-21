import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def metrics(model, val_loader):
  DEVICE = next(model.parameters()).device
  results = []
  model.eval()
  with torch.no_grad():
    for data, target in val_loader:
      output = model(data.to(DEVICE)).to('cpu')
      results += [(
          output.detach().numpy(),
          target.detach().to('cpu').numpy(),
      )]
  logits, targets = zip(*results)
  logits = np.concatenate(logits)
  # only needed for accuracy
  #predictions = np.argmax(logits, axis=1)
  targets = np.concatenate(targets)
  return dict(roc=roc_auc_score(targets, logits[:, 1]))