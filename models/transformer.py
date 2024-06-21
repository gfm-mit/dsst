import numpy as np
import torch

import models.base
from models.util import NoopAttention


class Decoder(torch.nn.Module):
  def __init__(self, n_features, device=None, causal=False):
      super(Decoder, self).__init__()
      decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=12,
        nhead=2,
        dim_feedforward=n_features,
        batch_first=True,
      )
      decoder_layer.self_attn = torch.nn.MultiheadAttention(
        embed_dim=12,
        num_heads=2,
        dropout=0.1,
        batch_first=True,
        bias=True)
      decoder_layer.multihead_attn = NoopAttention() # this is used on the memory
      self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=2)
      self.causal = causal
      self.device = device

  def forward(self, input):
    mask = None
    if self.causal:
      seq_len = input.shape[1]
      # tgt, src
      mask = torch.Tensor(np.tril(np.ones((seq_len, seq_len)), k=-1).astype(bool)).to(self.device)
    return self.decoder(input, memory=None, tgt_mask=mask)

class GetClassifierOutputs(torch.nn.Module):
  def __init__(self, kind=None):
    self.kind = kind
    assert self.kind in "meanmax max mean last".split()
    super(GetClassifierOutputs, self).__init__()

  def forward(self, input):
    if self.kind == "max":
      return torch.amax(input, dim=1)
    elif self.kind == "mean":
      nonzeros = torch.sum(torch.amax(input != 0, dim=2, keepdim=True), dim=1)
      return torch.sum(input, dim=1) / nonzeros
    elif self.kind == "meanmax":
      max = torch.amax(input, dim=1)
      nonzeros = torch.sum(torch.amax(input != 0, dim=2, keepdim=True), dim=1)
      mean = torch.sum(input, dim=1) / nonzeros
      return torch.cat([max, mean], dim=1)
    elif self.kind == "last":
      return input[:, -1, :]
    assert False, "unknown transformer aggregation kind: {}".format(self.kind)

class Transformer(models.base.SequenceBase):
  def __init__(self, n_layers=2, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    self.layers = n_layers
    super().__init__(device=device)

  def get_next_token_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        Decoder(n_features=12, device=self.device, causal=True),
    )
    model = model.to(self.device)
    return model

  def get_classifier_architecture(self, agg_kind=None):
    n_features = self.features
    if agg_kind == "meanmax":
      n_features *= 2
    model = torch.nn.Sequential(
        # b n c
        Decoder(n_features=12, device=self.device, causal=True), # TODO: try False
        GetClassifierOutputs(kind=agg_kind),
        torch.nn.BatchNorm1d(num_features=n_features),
        torch.nn.Linear(n_features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_next_token_parameters(self, **kwargs):
    return dict(
      scheduler='none',
      optimizer='sfadam',
      learning_rate=2e-2,
      momentum=0.9,
      warmup_step=3,
      max_epochs=200,
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      scheduler='none',
      optimizer='samadam',
      weight_decay=0,
      momentum=.9,
      conditioning_smoother=.999,
      agg_kind="max",

      max_epochs=5,
      warmup_epochs=5,

      learning_rate=1.3e-1, # stupid edge of stability!!
    )