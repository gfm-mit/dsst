import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

def evaluate(model, val_loader, task):
  if task == "next_token":
    return next_token_metrics(model, val_loader)
  else:
    return binary_classifier_metrics(model, val_loader)

def next_token_metrics(model, val_loader, verbose=True):
  DEVICE = next(model.parameters()).device
  results = []
  model.eval()
  with torch.no_grad():
    for data, target in val_loader:
      output = model(data.to(DEVICE)).to('cpu')
      results += [(
          output.detach().numpy()[:, :-1, :],
          data.detach().to('cpu').numpy()[:, 1:, :],
      )]
  predicted, data = zip(*results)
  predicted = np.concatenate(predicted)
  data = np.concatenate(data)

  if verbose:
    var = np.mean((data)**2, axis=(0, 1))
    mse = np.mean((predicted - data)**2, axis=(0, 1))
    r2 = np.mean(np.reshape(mse / var, [6, 2]), axis=1)
    with pd.option_context('display.float_format', '{:.1f}%'.format):
      print(pd.Series(100 * r2,
                      index="t v_mag2 a_mag2 dv_mag2 cw j_mag2".split(),
                      name="Verbose MSE components"
                      ))
  mse = float(np.mean((predicted - data)**2))
  return mse

def binary_classifier_metrics(model, val_loader):
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
  return float(roc_auc_score(targets, logits[:, 1]))

def get_combined_roc(model, test_loader, combine_fn=None):
  logits = []
  targets = []
  groups = []
  DEVICE = next(model.parameters()).device
  with torch.no_grad():
    for idx, (data, target, g) in enumerate(test_loader):
      #print(target)
      output = model(data.to(DEVICE)).to('cpu')
      logits += [output.detach().numpy()[:, 1]]
      targets += [target.detach().to('cpu').numpy()[:, 0]]
      groups += [g]
      if idx % 100 == 0 and idx > 0:
        print(f"metrics.get_combined_roc()[{idx}]")
  # TODO: why is this thing not working at all?
  logits = np.concatenate(logits)
  targets = np.concatenate(targets)
  groups = np.concatenate(groups)
  if combine_fn is not None:
    logits, targets = combine_fn(logits, targets, groups)
  return logits, targets