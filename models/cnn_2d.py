import torch
from einops.layers.torch import Rearrange

import models.base


class Cnn(models.base.Base):
  def __init__(self, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    super().__init__(device=device)

  def get_classifier_architecture(self):
    model = torch.nn.Sequential(
        # b c h w
        torch.nn.LayerNorm([28, 28]),
        torch.nn.Conv2d(
            in_channels=1,
            out_channels=self.features,
            kernel_size=5,
            stride=3,
            padding=3,
            bias=False,
        ),
        torch.nn.LayerNorm([10, 10]),
        torch.nn.Conv2d(
            in_channels=self.features,
            out_channels=self.features,
            kernel_size=5,
            stride=3,
            padding=3,
            bias=False,
        ),
        torch.nn.LayerNorm([4, 4]),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        Rearrange('b c 1 1 -> b c'),
        torch.nn.Linear(self.features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_parameters(self, **kwargs):
    return dict(
      scheduler='onecycle',
      weight_decay=0,
      momentum=0,
      conditioning_smoother=0.9,
      pct_start=0.0,

      max_epochs=10,
      min_epochs=0,

      learning_rate=5e-1, # stupid edge of stability!!
    )