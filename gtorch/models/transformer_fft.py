import numpy as np
import pandas as pd
import torch
from einops import rearrange

import gtorch.models.base
from gtorch.models.util import FourierAttention, NoopAttention

class Decoder(torch.nn.Module):
  def __init__(self, n_features):
      super(Decoder, self).__init__()
      decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=12,
        nhead=2,
        dim_feedforward=n_features,
      )
      decoder_layer.self_attn = FourierAttention()
      decoder_layer.multihead_attn = NoopAttention() # this is used on the memory
      self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=2)

  def forward(self, input):
    bnc_in = rearrange(input, 'b n c -> n b c')
    bnc_out = self.decoder(bnc_in, None)
    return bnc_out

class GetClassifierOutputs(torch.nn.Module):
  def forward(self, input):
    return input[-1, :, :]

class Transformer(gtorch.models.base.SequenceBase):
  def __init__(self, n_layers=2, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    self.layers = n_layers
    self.device = device
    super().__init__()

  def get_next_token_architecture(self, hidden_width='unused'):
    model = torch.nn.Sequential(
        # b n c
        Decoder(n_features=12),
    )
    model = model.to(self.device)
    return model

  def get_classifier_architecture(self, hidden_width='unused'):
    model = torch.nn.Sequential(
        # b n c
        Decoder(n_features=12),
        GetClassifierOutputs(),
        torch.nn.BatchNorm1d(num_features=self.features),
        torch.nn.Linear(self.features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_next_token_parameters(self, **kwargs):
    return dict(
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      #optimizer='adam',
      schedule='onecycle',
      weight_decay=0,
      momentum=1 - 1e-3,
      beta2=1 - 3e-2,
      pct_start=0.0,

      max_epochs=2,
      min_epochs=0,

      learning_rate=1e-1, # stupid edge of stability!!
      hidden_width=2,
    )

  def get_tuning_ranges(self):
    return dict(
        #nonce=np.arange(5),
        #learning_rate=np.geomspace(1e-5, 1e-0, 15),
        #weight_decay=np.geomspace(1e-8, 1e-4, 15),
        #pct_start=np.geomspace(0.01, .95, 15),
        #max_epochs=np.geomspace(5, 100, 15).astype(int),
        #momentum=1-np.geomspace(.1, 1e-5, 15),
        #beta2=1-np.geomspace(.5, .0001, 15),
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