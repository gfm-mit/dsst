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

  def get_causal_cnn(self):
    return torch.nn.Sequential(
        models.util.CausalConv1d(12, 12, kernel_size=2, dilation=1),
    )

  def get_next_token_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        self.get_causal_cnn(),
        Rearrange('b c n -> b n c'),
    )
    model = model.to(self.device)
    return model

  def get_classifier_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        self.get_causal_cnn(),
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
      max_epochs=15,
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      scheduler='warmup',
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=5,
      max_epochs=30,

      learning_rate=2e-2,
    )