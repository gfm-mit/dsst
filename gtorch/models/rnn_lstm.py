import numpy as np
import pandas as pd
import torch
from einops import rearrange

import gtorch.models.base
#from gtorch.models.util import OneCat, PrintCat

class GetHidden(torch.nn.Module):
  def forward(self, input):
    output, (hidden, gating) = input
    return rearrange(hidden, 'd b c -> b (d c)')

class GetOutput(torch.nn.Module):
  def forward(self, input):
    output, (hidden, gating) = input
    return output

class Rnn(gtorch.models.base.SequenceBase):
  def __init__(self, n_layers=2, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    self.layers = n_layers
    self.device = device
    super().__init__()

  def get_lstm(self):
    return torch.nn.LSTM(
      input_size=12,
      hidden_size=self.features,
      num_layers=2,
      batch_first=True)

  def get_next_token_architecture(self, hidden_width='unused'):
    model = torch.nn.Sequential(
        # b n c
        self.get_lstm(),
        GetOutput(),
    )
    model = model.to(self.device)
    return model

  def get_classifier_architecture(self, hidden_width='unused'):
    model = torch.nn.Sequential(
        # b n c
        self.get_lstm(),
        GetHidden(),
        torch.nn.BatchNorm1d(num_features=self.layers * self.features),
        torch.nn.Linear(self.layers * self.features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_next_token_parameters(self):
    optimizer = "adam"
    if optimizer == "adam":
      return dict(
        optimizer="adam",
        learning_rate=2e-1,
        momentum=1 - 1e-5,
        beta2=1 - 1e-5,
        weight_decay=2e-2, # probably bogus
        max_epochs=10, # 100 is too many
      )
    return dict(
      learning_rate=1e+1
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      #optimizer='adam',
      schedule='onecycle',
      weight_decay=0,
      momentum=1 - 1e-3,
      beta2=1 - 3e-2,
      pct_start=0.0,

      max_epochs=1,
      min_epochs=0,

      learning_rate=1e-1,
      hidden_width=2,
    )

  def get_tuning_ranges(self):
    return dict(
        #nonce=np.arange(5),
        #optimizer=['adam'] * 8 + ['sgd'] * 7,
        #learning_rate=np.geomspace(1e-1, 1e+2, 15),
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