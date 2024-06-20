import torch
from einops.layers.torch import Rearrange

import gtorch.models.base
from gtorch.models.util import PadWidthToMultiple


class Cnn(gtorch.models.base.Base):
  def __init__(self, n_features=128, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    super().__init__(device=device)

  def get_classifier_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        PadWidthToMultiple(8),
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