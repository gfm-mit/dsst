import numpy as np
import torch
from einops.layers.torch import Rearrange

import models.base

class Cnn(models.base.Base):
  def __init__(self, n_classes=2, n_inputs=12, device='cpu'):
    self.classes = n_classes
    self.inputs = n_inputs
    print(f"{self.inputs=}")
    super().__init__(device=device)

  def get_classifier_architecture(
      self, arch_kernel=None, arch_fft_width=None, arch_conv_width=None, arch_fft_length=None, arch_dropout=None, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        torch.nn.Conv1d(12, arch_fft_width, kernel_size=arch_kernel),
        torch.nn.SiLU(),
        models.util.FftBlock(trim=arch_fft_length),
        torch.nn.Dropout1d(arch_dropout),
        torch.nn.BatchNorm1d(num_features=arch_fft_width),
        torch.nn.Conv1d(arch_fft_width, arch_conv_width, 1),
        torch.nn.SiLU(),
        torch.nn.BatchNorm1d(num_features=arch_conv_width),
        torch.nn.Conv1d(arch_conv_width, arch_conv_width, 1),
        torch.nn.SiLU(),
        Rearrange('b c n -> b (c n)'),
        torch.nn.BatchNorm1d(num_features=arch_conv_width * arch_fft_length),
        torch.nn.Linear(arch_conv_width * arch_fft_length, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_parameters(self, **kwargs):
    return dict(
      scheduler='none',
      optimizer='prodigy',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=2,
      max_epochs=40,
      learning_rate=3e-3,

      arch_kernel=5,
      arch_fft_length=32,
      arch_fft_width=192,
      arch_conv_width=12,
      arch_dropout=0.1,
    )