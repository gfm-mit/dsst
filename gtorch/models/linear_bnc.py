import numpy as np
import pandas as pd
import torch
from einops.layers.torch import Rearrange

import gtorch.models.base


class Linear(gtorch.models.base.Base):
  def __init__(self, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    super().__init__(device=device)

  def get_classifier_architecture(self):
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

  def get_parameters(self, **kwargs):
    return dict(
      scheduler='none',
      optimizer='sgd',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      pct_start=0.0,
      max_epochs=5,
      min_epochs=0,

      learning_rate=3e-1, # stupid edge of stability!!
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