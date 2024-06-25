import numpy as np
import torch

class SwiGLU(torch.nn.Module):
  def forward(self, x):
    x, gate = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * x

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

class CausalConv1d(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dilation):
    super().__init__()
    self.padding = dilation * (kernel_size - 1)
    self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
    with torch.no_grad():
      self.conv.weight.data.zero_()
      self.conv.bias.data.zero_()

  def forward(self, x):
      x = self.conv(x)
      if self.padding > 0:
        x = x[:, :, :-self.padding]
      return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs