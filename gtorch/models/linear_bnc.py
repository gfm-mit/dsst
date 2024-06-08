import numpy as np
import pandas as pd
import torch
from einops.layers.torch import Rearrange

import gtorch.models.base
#from gtorch.models.util import OneCat, PrintCat


class Linear(gtorch.models.base.Base):
  def __init__(self, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    self.device = device
    super().__init__()

  def get_classifier_architecture(self, hidden_width='unused'):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        torch.nn.BatchNorm1d(num_features=12),
        torch.nn.Linear(self.features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_parameters(self):
    return dict(
      #optimizer='adam',
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
        learning_rate=np.geomspace(1e-1, 5e-1, 15),
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
          [model.state_dict()['4.bias'].numpy()[1]
          - model.state_dict()['4.bias'].numpy()[0]],
          model.state_dict()['4.weight'].numpy()[1]
          - model.state_dict()['4.weight'].numpy()[0]
        ])
      )
    return pd.Series(
      np.concatenate([
        model.state_dict()['4.bias'].numpy(),
        model.state_dict()['4.weight'].numpy().flatten(),
      ])
    )