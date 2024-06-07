#@title def get_jank_model(Linear):
import numpy as np
import pandas as pd
import torch
from einops import rearrange

import gtorch.models.base

class OneCat(torch.nn.Module):
  def forward(self, input):
    return torch.cat([torch.zeros([input.shape[0], 1]), input], axis=1)

class GetHidden(torch.nn.Module):
  def forward(self, input):
    output, (hidden, gating) = input
    return rearrange(hidden, 'd b c -> b (d c)')

class PrintCat(torch.nn.Module):
  def forward(self, input):
    print(input.shape)
    return input

class Rnn(gtorch.models.base.Base):
  def __init__(self, n_layers=2, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    self.layers = n_layers
    self.device = device
    super().__init__()

  def get_architecture(self, hidden_width='unused'):
    model = torch.nn.Sequential(
        # b n c
        torch.nn.LSTM(
          input_size=12,
          hidden_size=self.features,
          num_layers=2,
          batch_first=True),
        GetHidden(),
        torch.nn.BatchNorm1d(num_features=self.layers * self.features),
        torch.nn.Linear(self.layers * self.features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_parameters(self):
    return dict(
      solver='adam',
      schedule='onecycle',
      weight_decay=0,
      momentum=1 - 1e-3,
      beta2=1 - 3e-2,
      pct_start=0.0,

      max_epochs=10,
      min_epochs=0,

      learning_rate=1e-1, # stupid edge of stability!!
      hidden_width=2,
    )

  def get_tuning_ranges(self):
    return dict(
        #nonce=np.arange(5),
        #learning_rate=np.geomspace(1e-2, 1e-0, 15),
        #weight_decay=np.geomspace(1e-8, 1e-4, 15),
        #pct_start=np.geomspace(0.01, .95, 15),
        #max_epochs=np.geomspace(5, 100, 15).astype(int),
        #momentum=1-np.geomspace(.1, 1e-5, 35),
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