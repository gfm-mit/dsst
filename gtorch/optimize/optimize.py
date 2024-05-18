import numpy as np
import pandas as pd
import torch
import shutil
from pathlib import Path
import re
import matplotlib.pyplot as plt
import scipy
from tqdm.notebook import tqdm
import einops
from einops.layers.torch import Rearrange
from pathlib import Path
from sklearn.metrics import roc_auc_score

Path('./results/').mkdir(parents=True, exist_ok=True)

def balance_class_weights(target):
    w = torch.nn.functional.one_hot(target).float().mean(axis=0)
    w = 1 / w
    w = torch.clip(w, 0, 10)
    w = w / w.sum()
    return w

def optimize(epoch, model, optimizer, train_loader):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  #gradients = {}
  #
  #def hook(module, grad_input, grad_output):
  #    print(type(module), grad_input[0].shape, grad_output[0].shape)
  #    gradients[module] = torch.cat([
  #       grad_input[0],
  #       #torch.ones(grad_input[0].shape[0], 1).to(DEVICE),
  #       grad_output[0],
  #    ], axis=1)
  #    gradients[str(module) + "_in"] = grad_input[0].shape
  #    gradients[str(module) + "_out"] = grad_output[0].shape
  #
  #for layer in model.children():
  #  layer.register_backward_hook(hook)
  for data, target in train_loader:
    loader_has_batches = True
    optimizer.zero_grad()
    output = model(data.to(DEVICE))
    class_weights = balance_class_weights(target)
    #loss = torch.nn.functional.nll_loss(
    #    output, target.to(DEVICE), weight=class_weights.to(DEVICE))
    loss = torch.nn.functional.nll_loss(output, target[:, 0].to(DEVICE))
    loss.backward()
    #print("target", target[0])
    #for k, v in gradients.items():
    #  print(k, v)
    #print("output.grad", output.grad)
    #for k, v in model.named_parameters():
    #  print(k, v.grad)
    optimizer.step()
  assert loader_has_batches
  torch.save(model.state_dict(), './results/model.pth')
  print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
  return loss.item()

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
  predictions = np.argmax(logits, axis=1)
  targets = np.concatenate(targets)
  #return None, (predictions == targets).mean()
  # AUC is more useful than accuracy here.
  return None, roc_auc_score(targets, logits[:, 1])

class FakeOptimizer():
  def __init__(self, model):
    super(FakeOptimizer, self).__init__()
    layer_lookup = {
        k: str(v)
        for k, v in model.named_modules()
        if k
    }
    parameters = [
        (list(map(str, v.shape)), *k.split(".", maxsplit=1))
        for k, v in model.named_parameters()
    ]
    for s, l, p in parameters:
      print("{:>10}: {:10}<-{}".format(
          ", ".join(s[::-1]), p, layer_lookup.get(l, l)
      ))
  def zero_grad(self):
    pass
  def step(self):
    pass
  def state_dict(self):
    return {}