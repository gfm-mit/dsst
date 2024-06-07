#@title def get_jank_model(Linear):
import numpy as np
import pandas as pd
import torch
from einops.layers.torch import Rearrange

import gtorch.models.base

class PadCat(torch.nn.Module):
  def __init__(self, width=8):
    super().__init__()
    self.width = width

  def forward(self, input):
    n_h = int(np.ceil(input.shape[2] / self.width)) * self.width
    return torch.nn.functional.pad(input, (0, n_h - input.shape[2]))

class PrintCat(torch.nn.Module):
  def forward(self, input):
    print(input.shape)
    return input

class Cnn(gtorch.models.base.Base):
  def __init__(self, n_features=128, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    self.device = device
    super().__init__()

  def get_architecture(self, hidden_width='unused'):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        PadCat(8),
        Rearrange('b c (h w) -> b c h w', h=8),
        # separable 5x5: slow on CPU, might be fast on GPU
        torch.nn.Conv2d(
          12,
          self.features,
          kernel_size=(1, 5),
          stride=(1, 5)),
        torch.nn.BatchNorm2d(num_features=self.features),
        torch.nn.Conv2d(
          self.features,
          self.features,
          kernel_size=(5, 1),
          stride=(5, 1)),
        torch.nn.BatchNorm2d(num_features=self.features),
        torch.nn.AdaptiveMaxPool2d((8, 8)),
        # separable 5x5: slow on CPU, might be fast on GPU
        torch.nn.Conv2d(
          self.features,
          self.features,
          kernel_size=(1, 5),
          stride=(1, 5)),
        torch.nn.BatchNorm2d(num_features=self.features),
        torch.nn.Conv2d(
          self.features,
          self.features,
          kernel_size=(5, 1),
          stride=(5, 1)),
        torch.nn.BatchNorm2d(num_features=self.features),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Rearrange('b c 1 1 -> b c'),
        torch.nn.BatchNorm1d(num_features=self.features),
        torch.nn.Linear(self.features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
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

      learning_rate=1e-1, # stupid edge of stability!!
      hidden_width=2,
    )

  def get_tuning_ranges(self):
    return dict(
        #nonce=np.arange(5),
        #learning_rate=np.geomspace(1e-5, 1e-0, 15),
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