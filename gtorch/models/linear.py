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
  def __init__(self, classes=2):
    self.classes = classes
    super().__init__()

  def get_architecture(self, device='cpu', hidden_width='unused'):
    device = 'cpu' #TODO: jank!!!!
    if self.classes == 1:
      model = torch.nn.Sequential(
          Rearrange('b c 1 -> b c'),
          torch.nn.LayerNorm(normalized_shape=12),
          torch.nn.Linear(12, 1),
          OneCat(),
          torch.nn.LogSoftmax(dim=-1),
      )
    else:
      model = torch.nn.Sequential(
          Rearrange('b c 1 -> b c'),
          torch.nn.LayerNorm(normalized_shape=12),
          torch.nn.Linear(12, self.classes),
          torch.nn.LogSoftmax(dim=-1),
      )
    model = model.to(device)
    torch.nn.init.zeros_(model[1].weight)
    torch.nn.init.zeros_(model[1].bias)
    return model
  
  def get_parameters(self):
    return dict(
      weight_decay=1e-7,
      momentum=0.999,
      beta2=0.9,
      pct_start=0.0,

      max_epochs=10,
      min_epochs=0,

      learning_rate=1e-3,
      hidden_width=2,
    )

  def get_tuning_ranges(self):
    return dict(
        nonce=np.arange(35),
        #learning_rate=np.geomspace(1e-5, 1e1, 15),
        #weight_decay=np.geomspace(1e-8, 1e0, 35), 
        #pct_start=np.geomspace(0.005, .95, 35),
        #max_epochs=np.linspace(5, 15, 35).astype(int),
        #momentum=1-np.geomspace(.1, .0001, 15),
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