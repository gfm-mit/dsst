import torch
from einops.layers.torch import Rearrange
import re

import models.base
from pytorch_tcn import TCN

class swish(torch.nn.Module):
  def __init__(self):
    self.silu = torch.nn.SiLU()
  
  def forward(self, x):
    return self.silu(x)
class Cnn(models.base.SequenceBase):
  def __init__(self, n_classes=2, n_inputs=12, device='cpu'):
    self.classes = n_classes
    self.inputs = n_inputs
    super().__init__(device=device)

  def get_next_token_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        #models.util.SineProjection(self.inputs, kwargs['arch_width'], 1), # why is this 1?
        TCN(num_inputs=self.inputs,
            num_channels=[kwargs['arch_width']] * kwargs['arch_depth'],
            kernel_size=kwargs['arch_kernel'],
            dropout=kwargs['arch_dropout'],
            activation='selu',
            use_skip_connections=True),
        Rearrange('b c n -> b n c'),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(normalized_shape=kwargs['arch_width']),
        torch.nn.Linear(kwargs['arch_width'], self.inputs),
    )
    model = model.to(self.device)
    return model
  
  def translate_state_dict(self, next_token_state_dict):
    classifier_state_dict = {}
    for k, v in next_token_state_dict.items():
      if re.match("1[.]network[.]0[.]", k):
        print(f"saving param {k}")
        classifier_state_dict[k] = v
      else:
        print(f"not saving param {k}")
    return classifier_state_dict

  def get_classifier_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        TCN(num_inputs=self.inputs,
            num_channels=[kwargs['arch_width']] * kwargs['arch_depth'],
            kernel_size=kwargs['arch_kernel'],
            dropout=kwargs['arch_dropout'],
            activation='selu',
            use_skip_connections=True),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        torch.nn.BatchNorm1d(num_features=kwargs['arch_width']),
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
      max_epochs=20,
      learning_rate=1e-2,
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      scheduler='warmup', # TODO: try this with a scheduler
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=2,
      max_epochs=30,
      learning_rate=1e-3,

      arch_width=96,
      arch_kernel=3, # worse than 1, though
      arch_depth=1, # probably a fluke
      arch_dropout=0.05,
    )