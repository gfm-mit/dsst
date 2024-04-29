#@title def one_hyperparam(), many_hyperparams():
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

from gtorch.optimize.optimize import optimize, metrics

Path('./results').mkdir(parents=True, exist_ok=True)
def one_hyperparam(model, optimizer, scheduler, min_epochs, max_epochs, train_loader, val_loader):
  max_loss = 3
  resdict = dict(loss=np.nan, accuracy=np.nan)
  for x in range(max_epochs):
    #train_data = ContrastiveDataset('./TRAIN/', seed=str(np.random.randint(10)))
    #val_data = ContrastiveDataset('./TEST/', seed=str(np.random.randint(10)))
    # TODO: reshuffle data after each epoch, of course
    #train_loader = torch.utils.data.DataLoader(
    #    train_data, batch_size=batch_size_train, shuffle=False, collate_fn=collate_fn_padd)
    #val_loader = torch.utils.data.DataLoader(
    #    val_data, batch_size=batch_size_test, shuffle=False, collate_fn=collate_fn_padd)

    next_loss = optimize(x, model, optimizer, train_loader)
    scheduler.step()
    if next_loss > max_loss and x > min_epochs:
      break
    if np.isnan(next_loss):
      break
    max_loss = next_loss
    torch.cuda.empty_cache()
    resdict = dict(zip("loss accuracy".split(), metrics(model, val_loader)))

    torch.cuda.empty_cache()
  return resdict, model

def many_hyperparams(params, model_factory_fn, pretrained=False, train_loader=None, val_loader=None):
  model, _ = model_factory_fn(**{
      k: params[k]
      for k in "hidden_width".split()
      })

  if pretrained:
    network_state_dict = torch.load('sequence_pretrain.pth')
    model.load_state_dict(network_state_dict, strict=False)
  else:
    for layer in model.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

  optimizer = torch.optim.AdamW(model.parameters(),
                                lr=params["learning_rate"],
                                betas=[
                                    params["momentum"],
                                    params["beta2"],
                                ],
                                weight_decay=params["weight_decay"])
  scheduler = torch.optim.lr_scheduler.OneCycleLR(
      optimizer,
      max_lr=params["learning_rate"],
      steps_per_epoch=1,
      pct_start=params["pct_start"],
      epochs=int(params["max_epochs"]))

  torch.cuda.empty_cache()
  return one_hyperparam(model, optimizer, scheduler,
                        min_epochs=params["min_epochs"],
                        max_epochs=params["max_epochs"],
                        train_loader=train_loader,
                        val_loader=val_loader)