import numpy as np
import torch

class OnePadChannelsForBias(torch.nn.Module):
  def forward(self, input):
    return torch.cat([torch.zeros([input.shape[0], 1]), input], axis=1)

class PrintfModule(torch.nn.Module):
  def __init__(self, tag=""):
    super().__init__()
    self.tag = tag

  def forward(self, input):
    print(f"{self.tag=} {input.shape=}")
    return input

class ZeroPadLastDim(torch.nn.Module):
  def __init__(self, min_size=None, chunk_size=None):
    super().__init__()
    assert min_size is not None or chunk_size is not None
    self.min_size = min_size
    self.chunk_size = chunk_size

  def forward(self, input):
    if self.chunk_size is not None:
      min_size = np.ceil(float(input.shape[-1]) / self.chunk_size) * self.chunk_size
    else:
      min_size = self.min_size
    padding = min_size - input.shape[-1]
    if padding <= 0:
      return input
    else:
      return torch.nn.functional.pad(input, (0, padding))

class NoopAttention(torch.nn.Module):
    def forward(self, query, key, value,
                attn_mask=None,
                key_padding_mask=None,
                is_causal=None,
                need_weights=False):
        return query