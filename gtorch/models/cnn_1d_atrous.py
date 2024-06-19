import numpy as np
import pandas as pd
import torch
from einops.layers.torch import Rearrange

import gtorch.models.base
#from gtorch.models.util import OneCat, PrintCat


class Cnn(gtorch.models.base.Base):
  def __init__(self, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    super().__init__(device=device)

  def get_classifier_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        torch.nn.Conv1d(
          12,
          self.features,
          kernel_size=5,
          stride=5),
        torch.nn.BatchNorm1d(num_features=self.features),
        torch.nn.Conv1d(
          self.features,
          self.features,
          kernel_size=5,
          dilation=4),
        torch.nn.BatchNorm1d(num_features=self.features),
        torch.nn.Conv1d(
          self.features,
          self.features,
          kernel_size=5,
          padding=32,
          dilation=16),
        torch.nn.BatchNorm1d(num_features=self.features),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        torch.nn.BatchNorm1d(num_features=self.features),
        torch.nn.Linear(self.features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_parameters(self, **kwargs):
    return dict(
      scheduler='none',
      optimizer='samadam',
      weight_decay=0,
      momentum=0.5,
      conditioning_smoother=0.999,
      warmup_steps=5,

      max_epochs=400, # maybe 1200, actually
      min_epochs=0,

      learning_rate=1e-2, # stupid edge of stability!!
    )

  def get_tuning_ranges(self):
    return dict(
        #optimizer="samsgd sfsgd sgd".split(),
        #nonce=np.arange(5),
        #learning_rate=np.geomspace(3e-3, 3e-1, 15),
        #weight_decay=[0, 1e-5],
        #pct_start=np.geomspace(0.01, .95, 15),
        #max_epochs=np.geomspace(5, 100, 15).astype(int),
        momentum=[0, .5, .9, 0.999],
        #conditioning_smoother=1-np.geomspace(.5, .0001, 15),
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