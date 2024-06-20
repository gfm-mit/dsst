import torch
from einops import rearrange

import models.base

class GetNextTokenOutputs(torch.nn.Module):
  def forward(self, input):
    output, (hidden, gating) = input
    return rearrange(hidden, 'd b c -> b (d c)')

class GetClassifierOutputs(torch.nn.Module):
  def forward(self, input):
    output, (hidden, gating) = input
    return output

class Rnn(models.base.SequenceBase):
  def __init__(self, n_layers=2, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    self.features = n_features
    self.layers = n_layers
    super().__init__(device=device)

  def get_lstm(self):
    return torch.nn.LSTM(
      input_size=12,
      hidden_size=self.features,
      num_layers=2,
      batch_first=True)

  def get_next_token_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        self.get_lstm(),
        GetClassifierOutputs(),
    )
    model = model.to(self.device)
    return model

  def get_classifier_architecture(self):
    model = torch.nn.Sequential(
        # b n c
        self.get_lstm(),
        GetNextTokenOutputs(),
        torch.nn.BatchNorm1d(num_features=self.layers * self.features),
        # TODO: one linear layer should not be enough to parse the internal state of the RNN
        torch.nn.Linear(self.layers * self.features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_next_token_parameters(self):
    optimizer = "adam"
    if optimizer == "adam":
      return dict(
        optimizer="adam",
        learning_rate=2e-1,
        momentum=1 - 1e-5,
        conditioning_smoother=1 - 1e-5,
        weight_decay=2e-2, # probably bogus
        max_epochs=10, # 100 is too many
      )
    return dict(
      learning_rate=1e+1
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      scheduler='onecycle',
      weight_decay=0,
      momentum=1 - 1e-3,
      conditioning_smoother=1 - 3e-2,
      pct_start=0.0,

      max_epochs=10,
      min_epochs=0,

      learning_rate=1e-1,
    )