import torch
from einops.layers.torch import Rearrange

import models.base


class Cnn(models.base.Base):
  def __init__(self, n_classes=2, device='cpu'):
    self.classes = n_classes
    super().__init__(device=device)

  def get_classifier_architecture(self, arch_width=None, arch_dilation=None, arch_kernel_size=None):
    dilation = [int(d) for d in arch_dilation.split(',')]
    assert len(dilation) == 2
    kernels = [int(d) for d in arch_kernel_size.split(',')]
    assert len(kernels) == 3
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        torch.nn.Conv1d(
          12,
          arch_width,
          kernel_size=kernels[0],
          stride=kernels[0]),
        torch.nn.SiLU(),
        torch.nn.BatchNorm1d(num_features=arch_width),
        torch.nn.Conv1d(
          arch_width,
          arch_width,
          kernel_size=kernels[1],
          dilation=dilation[0]),
        torch.nn.SiLU(),
        torch.nn.BatchNorm1d(num_features=arch_width),
        torch.nn.Conv1d(
          arch_width,
          arch_width,
          kernel_size=kernels[2],
          padding=32,
          dilation=dilation[1]),
        torch.nn.SiLU(),
        torch.nn.BatchNorm1d(num_features=arch_width),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        torch.nn.BatchNorm1d(num_features=arch_width),
        torch.nn.Linear(arch_width, self.classes),
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
      warmup_epochs=5,
      max_epochs=30,

      learning_rate=1e-3,
      arch_width=192,
      arch_dilation='2,16',
      arch_kernel_size='5,5,4',
    )