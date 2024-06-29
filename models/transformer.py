import numpy as np
import torch
from einops.layers.torch import Rearrange
import re
from x_transformers import Decoder

import models.base
from models.util import PrintfModule

class DecoderWrapper(Decoder):
  def __init__(self, arch_width, arch_depth, arch_head, arch_dropout):
    super().__init__(
      dim=arch_width,
      depth=arch_depth,
      heads=arch_head,
      ff_dropout=arch_dropout,
      alibi_pos_bias=True,
    )

class Transformer(models.base.SequenceBase):
  def __init__(self, n_classes=2, n_inputs=12, device='cpu'):
    self.classes = n_classes
    self.inputs = n_inputs
    super().__init__(device=device)

  def get_next_token_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        models.util.SineProjection(self.inputs, kwargs['arch_width'], scale=1, axis=-1),
        DecoderWrapper(**kwargs),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(normalized_shape=kwargs['arch_width']),
        torch.nn.Linear(kwargs['arch_width'], self.inputs),
    )
    model = model.to(self.device)
    return model
  
  def translate_state_dict(self, next_token_state_dict):
    classifier_state_dict = {}
    for k, v in next_token_state_dict.items():
      if re.match("0[.]projection[.]*|1[.](final_norm|layers)[.]*", k):
        print(f"saving param {k}")
        classifier_state_dict[k] = v
      else:
        print(f"not saving param {k}")
    return classifier_state_dict

  def get_classifier_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        models.util.SineProjection(self.inputs, kwargs['arch_width'], scale=1, axis=-1),
        DecoderWrapper(**kwargs),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(normalized_shape=kwargs['arch_width']),
        Rearrange('b n c -> b c n'),
        # TODO: surely this can be improved?
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
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
      scheduler='warmup',
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=5,
      max_epochs=20,
      learning_rate=6e-3,

      arch_depth=1,
      arch_width=96,
      arch_dropout=0.05,
      arch_head=4,
    )