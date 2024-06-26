import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
import re

import models.base
from models.util import NoopAttention


class Decoder(torch.nn.Module):
  def __init__(self, arch_width, arch_ff_width, arch_depth, arch_head, arch_dropout, causal=False):
      super(Decoder, self).__init__()
      decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=arch_width,
        nhead=arch_head,
        dim_feedforward=arch_ff_width,
        batch_first=True,
      )
      decoder_layer.self_attn = torch.nn.MultiheadAttention(
        embed_dim=arch_width,
        num_heads=arch_head,
        dropout=arch_dropout,
        batch_first=True,
        bias=True)
      decoder_layer.multihead_attn = NoopAttention() # this is used on the memory
      self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=arch_depth)
      self.causal = causal

  def forward(self, input):
    mask = None
    if self.causal:
      seq_len = input.shape[1]
      # tgt, src
      mask = torch.Tensor(np.tril(np.ones((seq_len, seq_len)), k=-1).astype(bool)).to(input.device)
    return self.decoder(input, memory=None, tgt_mask=mask)

class Transformer(models.base.SequenceBase):
  def __init__(self, n_classes=2, n_inputs=12, device='cpu'):
    self.classes = n_classes
    self.inputs = n_inputs
    super().__init__(device=device)

  def get_next_token_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        Decoder(causal=True, **kwargs), # TODO: try False
        torch.nn.SiLU(),
        torch.nn.LayerNorm(normalized_shape=kwargs['arch_width']),
        torch.nn.Linear(kwargs['arch_width'], self.inputs),
    )
    model = model.to(self.device)
    return model
  
  def translate_state_dict(self, next_token_state_dict):
    classifier_state_dict = {}
    for k, v in next_token_state_dict.items():
      if re.match("0[.](weight|bias)_[ih][ih]_l\d", k):
        kk = k.replace("1.residual.", "") # unused
        print(f"saving param {k=} {kk=}")
        classifier_state_dict[kk] = v
    return classifier_state_dict

  def get_classifier_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        Decoder(causal=True, **kwargs), # TODO: try False
        Rearrange('b n c -> b c n'),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        # TODO: why does this work better than adding more layers?  and why is taking the last hidden state not enough?
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

      arch_depth=2,
      arch_width=12,
      arch_ff_width=12,
      arch_dropout=0.1,
      arch_head=2,
    )