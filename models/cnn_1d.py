import torch
from einops.layers.torch import Rearrange

import models.base
import models.util


class Cnn(models.base.Base):
  def __init__(self, n_features=12, n_classes=2, device='cpu'):
    self.classes = n_classes
    n_features = n_features
    super().__init__(device=device)

  def get_classifier_architecture(self, n_features=None, kernel_split=None, activation=None):
    kernels = [int(k) for k in kernel_split.split(',')]
    assert len(kernels) == 3
    if activation == "relu":
      activation = torch.nn.ReLU
    elif activation == "gelu":
      activation = torch.nn.GELU
    elif activation == "none":
      activation = torch.nn.Identity
    elif activation == "swish":
      activation = torch.nn.SiLU
    model = torch.nn.Sequential(
        # b n c
        Rearrange('b n c -> b c n'),
        torch.nn.Conv1d(
          12,
          n_features,
          kernel_size=kernels[0],
          stride=kernels[0]),
        activation(),
        torch.nn.BatchNorm1d(num_features=n_features),
        torch.nn.Conv1d(
          n_features,
          n_features,
          kernel_size=kernels[1],
          stride=kernels[1]),
        activation(),
        torch.nn.BatchNorm1d(num_features=n_features),
        torch.nn.Conv1d(
          n_features,
          n_features,
          kernel_size=kernels[2],
          stride=kernels[2]),
        activation(),
        torch.nn.BatchNorm1d(num_features=n_features),
        #PrintfModule('After 64x down'),
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        torch.nn.BatchNorm1d(num_features=n_features),
        torch.nn.Linear(n_features, self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_parameters(self, **kwargs):
    return dict(
      scheduler='warmup',
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=5,

      max_epochs=50,

      learning_rate=1e-2,

      n_features=32,
      kernel_split='2,2,16',
      activation="swish",
    )