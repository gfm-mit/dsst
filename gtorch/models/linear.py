#@title def get_jank_model(Linear):
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
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import gtorch.models.base

class NegCat(torch.nn.Module):
  def forward(self, input):
    #return torch.cat([input, -input, torch.ones([input.shape[0], 1, 1])], axis=1)
    return torch.cat([input, -input], axis=1)

class OneCat(torch.nn.Module):
  def forward(self, input):
    return torch.cat([torch.zeros([input.shape[0], 1]), input], axis=1)

class PrintCat(torch.nn.Module):
  def forward(self, input):
    print(input[:, :, 0].T)
    return input

class Linear(gtorch.models.base.Base):
  def __init__(self, n_features=12, n_classes=2):
    self.classes = n_classes
    self.features = n_features
    super().__init__()

  def get_architecture(self, device='cpu', hidden_width='unused'):
    device = 'cpu' #TODO: jank!!!!
    if self.classes == 1:
      model = torch.nn.Sequential(
          Rearrange('b c 1 -> b c'),
          torch.nn.BatchNorm1d(num_features=12),
          torch.nn.Linear(self.features, 1),
          OneCat(),
          torch.nn.LogSoftmax(dim=-1),
      )
    else:
      model = torch.nn.Sequential(
          Rearrange('b c 1 -> b c'),
          torch.nn.BatchNorm1d(num_features=12),
          torch.nn.Linear(self.features, self.classes),
          torch.nn.LogSoftmax(dim=-1),
      )
    model = model.to(device)
    torch.nn.init.zeros_(model[1].weight)
    torch.nn.init.zeros_(model[1].bias)
    return model
  
  def get_parameters(self):
    return dict(
      #solver='adam',
      schedule='onecycle',
      weight_decay=0,
      momentum=0,
      beta2=0.9,
      pct_start=0.0,

      max_epochs=10,
      min_epochs=0,

      learning_rate=3e-1, # stupid edge of stability!!
      hidden_width=2,
    )

  def get_tuning_ranges(self):
    return dict(
        #nonce=np.arange(5),
        #learning_rate=np.geomspace(1e-1, 5e-1, 15),
        #weight_decay=np.geomspace(1e-8, 1e-4, 15), 
        #pct_start=np.geomspace(0.01, .95, 15),
        #max_epochs=np.geomspace(5, 100, 15).astype(int),
        #momentum=1-np.geomspace(.1, 1e-5, 15),
        #beta2=1-np.geomspace(.5, .0001, 15),
    )

  def get_coefficients(self, model):
    if self.classes == 2:
      return pd.Series(
        np.concatenate([
          [model.state_dict()['2.bias'].numpy()[1]
          - model.state_dict()['2.bias'].numpy()[0]],
          model.state_dict()['2.weight'].numpy()[1]
          - model.state_dict()['2.weight'].numpy()[0]
        ])
      )
    return pd.Series(
      np.concatenate([
        model.state_dict()['1.bias'].numpy(),
        model.state_dict()['1.weight'].numpy().flatten(),
      ])
    )