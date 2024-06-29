import numpy as np
import torch
from einops.layers.torch import Rearrange
import re

import models.base
from models.util import NoopAttention


class RelativeMultiheadAttention(torch.nn.MultiheadAttention):
  def __init__(self, embed_dim, num_heads, decay_rate=1.0, content_scale=1.0, **kwargs):
    super().__init__(embed_dim, num_heads, **kwargs)
    self.decay_rate = decay_rate
    self.content_scale = content_scale

  def forward(self, query, key, value, **kwargs):
    kwargs['need_weights'] = True
    attn_output, attn_weights = super().forward(query, key, value, **kwargs)
    positions = torch.arange(query.shape[1], device=query.device)
    rel_pos = torch.exp(-torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0)) / self.decay_rate)
    new_weights = torch.nn.functional.softmax(attn_weights * self.content_scale + rel_pos.unsqueeze(0), dim=-1)
    return torch.bmm(new_weights, value), new_weights

class Decoder(torch.nn.Module):
  def __init__(self, arch_width, arch_ff_width, arch_depth, arch_head, arch_dropout, arch_decay, arch_content_scale,
               arch_next_width='unused', arch_next_depth='unused', causal=False):
      super(Decoder, self).__init__()
      decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=arch_width,
        nhead=arch_head,
        dim_feedforward=arch_ff_width,
        batch_first=True,
      )
      attn_args = dict(
        embed_dim=arch_width,
        num_heads=arch_head,
        dropout=arch_dropout,
        batch_first=True,
        bias=True)
      if arch_decay > 0:
        decoder_layer.self_attn = RelativeMultiheadAttention(decay_rate=arch_decay, content_scale=arch_content_scale, **attn_args)
      else:
        decoder_layer.self_attn = torch.nn.MultiheadAttention(**attn_args)
      decoder_layer.multihead_attn = NoopAttention() # this is used on the memory
      self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=arch_depth)
      self.causal = causal

  def forward(self, input):
    #print(f"{input.shape=}")
    mask = None
    if self.causal:
      seq_len = input.shape[1]
      # tgt, src
      mask = torch.Tensor(np.tril(np.ones((seq_len, seq_len)), k=-1).astype(bool)).to(input.device)
    #print(f"{mask.shape=}")
    return self.decoder(input, memory=None, tgt_mask=mask)
  
class TokenResid(torch.nn.Module):
    def __init__(self, arch_next_width, arch_next_depth, arch_width):
      super().__init__()
      self.arch_next_width = arch_next_width
      self.arch_next_depth = arch_next_depth
      self.arch_width = arch_width
      self.stack = self.get_outer_stack()
    
    def get_inner_stack(self):
      return models.util.ResidualBlock(torch.nn.Sequential(
        torch.nn.LayerNorm(normalized_shape=self.arch_next_width),
        torch.nn.Linear(self.arch_next_width, self.arch_next_width),
        torch.nn.SiLU()))
    
    def get_outer_stack(self):
      return models.util.ResidualBlock(torch.nn.Sequential(
        torch.nn.SiLU(),
        torch.nn.LayerNorm(normalized_shape=self.arch_width),
        torch.nn.Linear(self.arch_width, self.arch_next_width),
        torch.nn.SiLU(),
        *[self.get_inner_stack() for _ in range(self.arch_next_depth)],
        torch.nn.LayerNorm(normalized_shape=self.arch_next_width),
        torch.nn.Linear(self.arch_next_width, self.arch_width),
        torch.nn.SiLU(),
      ))
    
    def forward(self, input):
      if self.arch_next_width == 0:
        return input
      return self.stack(input)

class Transformer(models.base.SequenceBase):
  def __init__(self, n_classes=2, n_inputs=12, device='cpu'):
    self.classes = n_classes
    self.inputs = n_inputs
    super().__init__(device=device)

  def get_next_token_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        models.util.SineProjection(self.inputs, kwargs['arch_width'], scale=1, axis=-1), # why is scale 1?
        Decoder(causal=True, **kwargs), # TODO: try False
        torch.nn.SiLU(),
        #TokenResid(
        #  arch_next_depth=kwargs['arch_next_depth'],
        #  arch_next_width=kwargs['arch_next_width'],
        #  arch_width=kwargs['arch_width']
        #  ) if kwargs['arch_next_width'] > 0 else torch.nn.Identity(),
        torch.nn.LayerNorm(normalized_shape=kwargs['arch_width']),
        torch.nn.Linear(kwargs['arch_width'], self.inputs),
    )
    model = model.to(self.device)
    return model
  
  def translate_state_dict(self, next_token_state_dict):
    classifier_state_dict = {}
    for k, v in next_token_state_dict.items():
      if re.match("0.projection.*|1.decoder.*", k):
        kk = k.replace("1.residual.", "") # unused
        print(f"saving param {k=} {kk=}")
        classifier_state_dict[kk] = v
    return classifier_state_dict

  def get_classifier_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        models.util.SineProjection(self.inputs, kwargs['arch_width'], scale=1, axis=-1), # why is scale 1?
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
      warmup_epochs=5,
      max_epochs=25,
      learning_rate=2e-2,
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      # this gets like 84% AUC, surprisingly enough
      scheduler='warmup', # TODO: try this with a scheduler
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=2,
      max_epochs=50,
      learning_rate=2e-3,

      arch_depth=1,
      arch_width=192,
      arch_ff_width=12,
      arch_dropout=0.05,
      arch_head=4,

      #arch_next_width=8,
      #arch_next_depth=0,
      arch_decay=3,
      arch_content_scale=3e-2,
    )