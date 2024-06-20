import torch
from einops.layers.torch import Rearrange

import models.base
from models.util import PrintfModule, ZeroPadLastDim


class Cnn(models.base.Base):
  def __init__(self, n_features=128, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    super().__init__(device=device)

  def get_classifier_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        #PrintfModule('DataLoader'), # 1000 128 12
        Rearrange('b n c -> b c n'),
        ZeroPadLastDim(chunk_size=8),
        Rearrange('b c (h w) -> b c h w', h=8),
        #PrintfModule('4D'), # 1000 12 8 16
        # separable 5x5: slow on CPU, might be fast on GPU
        torch.nn.Conv2d(
          12,
          self.features,
          kernel_size=(1, 4),
          stride=(1, 4)),
        torch.nn.BatchNorm2d(num_features=self.features),
        torch.nn.Conv2d(
          self.features,
          self.features,
          kernel_size=(4, 1),
          stride=(4, 1)),
        torch.nn.BatchNorm2d(num_features=self.features),
        #PrintfModule('Separable'), # 1000 12 8 16
        Rearrange('b c h w -> b c (h w)'),
        torch.nn.Conv1d(
          self.features,
          self.features,
          kernel_size=4,
          stride=4),
        #PrintfModule('Conv1D'), # 1000 12 8 16
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

      max_epochs=100,
      min_epochs=0,

      learning_rate=1e-2,  # not tuned
    )