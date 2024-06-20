import torch
from einops.layers.torch import Rearrange

import gtorch.models.base
from gtorch.models.util import ZeroPadLastDim


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
        ZeroPadLastDim(min_size=32), # bug in device=mps implementation, only
        torch.nn.AdaptiveMaxPool1d(32),
        torch.nn.Conv1d(
          self.features,
          self.features,
          kernel_size=5,
          stride=5),
        torch.nn.BatchNorm1d(num_features=self.features),
        torch.nn.Conv1d(
          self.features,
          self.features,
          kernel_size=5,
          stride=5),
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

      max_epochs=80,
      min_epochs=0,

      learning_rate=1e-2,  # not tuned
    )