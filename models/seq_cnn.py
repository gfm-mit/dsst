import torch
from einops.layers.torch import Rearrange

import models.base
from models.util import PrintfModule, CausalConv1d

class Cnn(models.base.SequenceBase):
  def __init__(self, n_layers=2, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    self.layers = n_layers
    super().__init__(device=device)

  def get_next_token_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        models.util.CausalConv1d(12, 12, kernel_size=2, dilation=1),
        Rearrange('b c n -> b n c'),
    )
    model = model.to(self.device)
    return model

  def get_classifier_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        torch.nn.Conv1d(12, 12, 1),
        #PrintfModule("after rearrange"),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        torch.nn.BatchNorm1d(num_features=12),
        # TODO: one linear layer should not be enough to parse the internal state of the RNN
        torch.nn.Linear(12, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_next_token_parameters(self):
    return dict(
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      scheduler='warmup',
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=3,
      max_epochs=10,

      learning_rate=1e-3,
      arch_width=192,
      arch_dilation='2,16',
      arch_kernel_size='5,5,4',
    )