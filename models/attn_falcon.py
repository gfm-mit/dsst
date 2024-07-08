import numpy as np
import torch
from einops.layers.torch import Rearrange
import re
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconConfig, build_alibi_tensor
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import warnings

import models.base
from models.util import PrintfModule

class DecoderWrapper(torch.nn.Module):
  def __init__(self, arch_width, arch_depth, arch_head, arch_dropout, device=None, arch_mask=False):
    super().__init__()
    self.device = device
    self.num_heads = arch_head
    config = FalconConfig(
      hidden_size=arch_width,
      ffn_hidden_size=arch_width,
      num_attention_heads=arch_head,
      num_hidden_layers=arch_depth,
      hidden_dropout=arch_dropout,
      attention_dropout=arch_dropout,
      new_decoder_architecture=True,
      cross_attention_dim=0,
      alibi=True,
    )
    config._attn_implementation = "sdpa"
    self.core = FalconDecoderLayer(config).to(device)
    self.mask_converter = AttentionMaskConverter(is_causal=True)
    self.mask = arch_mask
  
  def forward(self, x):
    batch_size, seq_length, width = x.shape
    mask = self.mask_converter.to_causal_4d(
        batch_size=x.shape[0],
        query_length=x.shape[1],
        key_value_length=x.shape[1],
        dtype=x.dtype,
        device=self.device
    )
    if self.mask:
      alibi_mask = torch.amax(x != 0, dim=2)
      unused_mask = ~torch.einsum("bi,bj->bij", alibi_mask, alibi_mask).to(self.device).unsqueeze(1)
      mask = mask.masked_fill(unused_mask, torch.finfo(mask.dtype).min)
    else:
      alibi_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=self.device)
    alibi = build_alibi_tensor(alibi_mask, self.num_heads, dtype=x.dtype).to(self.device)
    output, = self.core(x, alibi=alibi, attention_mask=mask)
    return output

class Transformer(models.base.SequenceBase):
  def __init__(self, n_classes=2, n_inputs=12, device='cpu'):
    self.classes = n_classes
    self.inputs = n_inputs
    super().__init__(device=device)
    # dammit, torch ecosystem, this is terrible
    warnings.simplefilter(action='ignore', category=FutureWarning)

  def get_next_token_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        models.util.SineProjection(self.inputs, kwargs['arch_width'], axis=-1, scale=1, preserve_zeros=True),
        DecoderWrapper(device=self.device, **kwargs),
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
        models.util.SineProjection(self.inputs, kwargs['arch_width'], axis=-1, scale=1, preserve_zeros=True),
        DecoderWrapper(device=self.device, **kwargs),
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
      warmup_epochs=8,
      max_epochs=12,
      batch=125,
      learning_rate=3e-2,
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
      learning_rate=2e-2,

      arch_depth=1,
      arch_width=24,
      arch_dropout=0.05,
      arch_head=4,
    )