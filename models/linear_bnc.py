import numpy as np
import pandas as pd
import torch
from einops.layers.torch import Rearrange

import models.base


class Linear(models.base.Base):
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
      scheduler='warmup',
      optimizer='samsgd', # maybe adam doesn't have time to warm up?
      weight_decay=0,
      momentum=0.999,
      conditioning_smoother=0.999,
      max_epochs=10,
      warmup_epochs=5,

      learning_rate=7e-2,
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