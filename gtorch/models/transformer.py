import numpy as np
import pandas as pd
import torch

import gtorch.models.base
from gtorch.models.util import NoopAttention


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
  def forward(self, input):
    return torch.amax(input, dim=1)
    return input[:, -1, :]

class Transformer(gtorch.models.base.SequenceBase):
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

  def get_classifier_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        Decoder(n_features=12, device=self.device, causal=True), # TODO: try False
        GetClassifierOutputs(),
        torch.nn.BatchNorm1d(num_features=self.features),
        torch.nn.Linear(self.features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_next_token_parameters(self, **kwargs):
    return dict(
      optimizer='adam',
      scheduler=None,
      learning_rate=1e-2,
      momentum=0.9,
      max_epochs=100,
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      scheduler='none',
      optimizer='samadam',
      weight_decay=0,
      momentum=.9,
      conditioning_smoother=.999,
      warmup_steps=5,

      max_epochs=20,
      min_epochs=5,

      learning_rate=1.3e-1, # stupid edge of stability!!
    )

  def get_tuning_ranges(self):
    return dict(
        #nonce=np.arange(5),
        learning_rate=np.geomspace(7e-2, 5e-1, 10),
        #weight_decay=np.geomspace(1e-8, 1e-4, 15),
        #pct_start=np.geomspace(0.01, .95, 15),
        #max_epochs=np.geomspace(5, 100, 15).astype(int),
        #momentum=[0, 0.5, 0.9, 0.999],
        #conditioning_smoother=1-np.geomspace(.5, .0001, 15),
    )

  def get_coefficients(self, model):
    if self.classes == 2:
      return pd.Series(
        np.concatenate([
          [model.state_dict()['2.bias'].numpy()[1]
          - model.state_dict()['2.bias'].numpy()[0]],
          model.state_dict()['2.weight'].numpy()[1]
          - model.state_dict()['2.weight'].numpy()[0]
        ])
      )
    return pd.Series(
      np.concatenate([
        model.state_dict()['1.bias'].numpy(),
        model.state_dict()['1.weight'].numpy().flatten(),
      ])
    )