import torch
from einops.layers.torch import Rearrange

import models.base


class Cnn(models.base.Base):
  def __init__(self, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    n_features = n_features
    super().__init__(device=device)

  def get_classifier_architecture(self, n_features=12):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        torch.nn.Conv1d(
          12,
          n_features,
          kernel_size=4,
          stride=4),
        torch.nn.BatchNorm1d(num_features=n_features),
        torch.nn.Conv1d(
          n_features,
          n_features,
          kernel_size=4,
          stride=4),
        torch.nn.BatchNorm1d(num_features=n_features),
        torch.nn.Conv1d(
          n_features,
          n_features,
          kernel_size=4,
          stride=4),
        torch.nn.BatchNorm1d(num_features=n_features),
        #PrintfModule('After 64x down'),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        torch.nn.BatchNorm1d(num_features=n_features),
        torch.nn.Linear(n_features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_parameters(self, **kwargs):
    return dict(
      scheduler='warmup',
      optimizer='samadam',
      weight_decay=0,
      momentum=0,
      conditioning_smoother=0.999,
      warmup_epochs=5,

      max_epochs=80,

      learning_rate=1.5e-2,

      n_features=48,
    )