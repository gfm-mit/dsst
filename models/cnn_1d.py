import torch
from einops.layers.torch import Rearrange

import models.base
import models.util


class Cnn(models.base.Base):
  def __init__(self, arch_width=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    arch_width = arch_width
    super().__init__(device=device)

  def get_classifier_architecture(self, arch_width=None, arch_kernel_list=None, arch_activation=None):
    kernels = [int(k) for k in arch_kernel_list.split(',')]
    assert len(kernels) == 3
    if arch_activation == "relu":
      arch_activation = torch.nn.ReLU
    elif arch_activation == "gelu":
      arch_activation = torch.nn.GELU
    elif arch_activation == "none":
      arch_activation = torch.nn.Identity
    elif arch_activation == "swish":
      arch_activation = torch.nn.SiLU
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        torch.nn.Conv1d(
          12,
          arch_width,
          kernel_size=kernels[0],
          stride=kernels[0]),
        arch_activation(),
        torch.nn.BatchNorm1d(num_features=arch_width),
        torch.nn.Conv1d(
          arch_width,
          arch_width,
          kernel_size=kernels[1],
          stride=kernels[1]),
        arch_activation(),
        torch.nn.BatchNorm1d(num_features=arch_width),
        torch.nn.Conv1d(
          arch_width,
          arch_width,
          kernel_size=kernels[2],
          stride=kernels[2]),
        arch_activation(),
        torch.nn.BatchNorm1d(num_features=arch_width),
        #PrintfModule('After 64x down'),
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

      max_epochs=50,

      learning_rate=1e-2,

      arch_width=32,
      arch_kernel_list='2,2,16',
      arch_activation="swish",
    )