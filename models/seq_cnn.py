from collections import OrderedDict
import torch
from einops.layers.torch import Rearrange

import models.base
from models.util import PrintfModule, CausalConv1d

class Cnn(models.base.SequenceBase):
  def __init__(self, n_classes=2, n_inputs=12, device='cpu'):
    self.classes = n_classes
    self.inputs = n_inputs
    print(f"{self.inputs=}")
    super().__init__(device=device)

  def get_causal_cnn(self, arch_width):
    #return torch.nn.Sequential(
    #    models.util.CausalConv1d(self.inputs, arch_width, kernel_size=1, dilation=1),
    #)
    return models.util.CausalConv1d(self.inputs, arch_width, kernel_size=1, dilation=1)

  def get_next_token_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        models.util.ResidualBlock(torch.nn.Sequential(
          self.get_causal_cnn(**kwargs),
          torch.nn.SiLU(),
          torch.nn.BatchNorm1d(num_features=kwargs['arch_width']),
          torch.nn.Dropout(0.1),
          torch.nn.Conv1d(kwargs['arch_width'], self.inputs, kernel_size=1),
        )),
        Rearrange('b c n -> b n c'),
    )
    model = model.to(self.device)
    return model

  def get_classifier_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        self.get_causal_cnn(**kwargs),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        torch.nn.BatchNorm1d(num_features=kwargs['arch_width']),
        # TODO: one linear layer should not be enough to parse the internal state of the RNN
        torch.nn.Linear(kwargs['arch_width'], self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_next_token_parameters(self):
    return dict(
      scheduler='warmup',
      optimizer='samadam',
      warmup_epochs=2,
      max_epochs=2,
      arch_width=24,
      learning_rate=1e-3,
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      scheduler='warmup',
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=5,
      max_epochs=2,
      learning_rate=2e-2,

      arch_width=24,
    )