import torch
from einops.layers.torch import Rearrange

import models.base


class Cnn(models.base.Base):
  def __init__(self, n_classes=2, device='cpu'):
    self.classes = n_classes
    super().__init__(device=device)

  def get_classifier_architecture(self, arch_channels=12):
    model = torch.nn.Sequential(
        # b c h w
        torch.nn.LayerNorm([28, 28]),
        torch.nn.Conv2d(
            in_channels=1,
            out_channels=arch_channels,
            kernel_size=5,
            stride=3,
            padding=3,
            bias=False,
        ),
        torch.nn.LayerNorm([10, 10]),
        torch.nn.Conv2d(
            in_channels=arch_channels,
            out_channels=arch_channels,
            kernel_size=5,
            stride=3,
            padding=3,
            bias=False,
        ),
        torch.nn.LayerNorm([4, 4]),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        Rearrange('b c 1 1 -> b c'),
        torch.nn.Linear(arch_channels, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_parameters(self, **kwargs):
    return dict(
      scheduler='warmup',
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,

      max_epochs=15,
      warmup_epochs=5,

      learning_rate=1e-1,
      arch_channels=64,
    )