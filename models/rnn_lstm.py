import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange

import models.base
from models.util import PrintfModule, CausalConv1d

class GetLastStepHiddenState(torch.nn.Module):
  def forward(self, input):
    output, (hidden, gating) = input
    return rearrange(hidden, 'd b c -> b (d c)')

class GetNextStepOutputs(torch.nn.Module):
  def forward(self, input):
    output, (hidden, gating) = input
    return output

class Rnn(models.base.SequenceBase):
  def __init__(self, n_classes=2, n_inputs=12, device='cpu'):
    self.classes = n_classes
    self.inputs = n_inputs
    super().__init__(device=device)

  def get_lstm(self, arch_width, arch_depth):
    return torch.nn.LSTM(
      input_size=self.inputs,
      hidden_size=int(arch_width),
      num_layers=arch_depth,
      batch_first=True)

  def get_next_token_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        torch.nn.Identity(),
        self.get_lstm(**kwargs),
        GetNextStepOutputs(),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(normalized_shape=kwargs['arch_width']),
        torch.nn.Linear(kwargs['arch_width'], self.inputs),
    )
    model = model.to(self.device)
    return model
  
  def translate_state_dict(self, next_token_state_dict):
    classifier_state_dict = {}
    for k, v in next_token_state_dict.items():
      if "causal_conv" in k or "projection" in k:
        kk = k.replace("1.residual.", "")
        classifier_state_dict[kk] = v
    return classifier_state_dict

  def get_classifier_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        self.get_lstm(**kwargs),
        GetNextStepOutputs(),
        Rearrange('b n c -> b c n'),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        #torch.nn.BatchNorm1d(num_features=kwargs['arch_width']),
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
      max_epochs=10,
      learning_rate=3e-2,
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      scheduler='warmup', # TODO: try this with a scheduler
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=2,
      max_epochs=50,
      learning_rate=3e-3,

      arch_width=48,
      arch_depth=2,
    )