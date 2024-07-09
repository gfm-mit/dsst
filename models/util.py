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

class FftBlock(torch.nn.Module):
  def __init__(self, trim=0):
    super().__init__()
    self.trim = trim

  def forward(self, x):
    fft = torch.fft.rfft(x, axis=2).abs()
    if self.trim != 0:
      fft = fft[:, :, :self.trim]
    return fft

class CausalConv1d(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dilation, downsample=None):
    super().__init__()
    dilation = 1 if downsample else dilation
    stride = 2 if downsample else 1
    self.padding = dilation * (kernel_size - 1)
    self.causal_conv = torch.nn.Conv1d(
       in_channels, out_channels, kernel_size,
       padding=self.padding,
       dilation=dilation,
       stride=stride)

  def forward(self, x):
      x = self.causal_conv(x)
      if self.padding > 0:
        x = x[:, :, :-self.padding]
      return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.residual = module

    def forward(self, inputs):
        return self.residual(inputs) + inputs

class ResidualBlock2(torch.nn.Module):
    def __init__(self, module1, module2):
        super().__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, inputs):
        output1 = self.module1(inputs)
        output2 = self.module2(inputs)
        return output1 + output2[:, :, :output1.shape[2]]

class SineProjection(torch.nn.Module):
  def __init__(self, in_width, out_width, scale=1, axis=-2, preserve_zeros=False):
    super().__init__()
    if axis == -2:
      self.projection = torch.nn.Conv1d(in_width, out_width, kernel_size=1, bias=not preserve_zeros)
    elif axis == -1:
      self.projection = torch.nn.Linear(in_width, out_width, bias=not preserve_zeros)
    else:
      assert False, f"SineProjection.{axis=}"
    self.projection.requires_grad_(False)
    self.scale = scale

  def forward(self, x):
    return torch.sin(self.scale*self.projection(x))

class SoftmaxAgg(torch.nn.Module):
  def __init__(self, dim, ff_width, norm=False):
    assert ff_width is not None
    super().__init__()
    if ff_width:
      self.fc1 = torch.nn.Linear(dim, ff_width)
      if norm:
        self.norm = torch.nn.LayerNorm(ff_width)
      else:
        self.norm = torch.nn.Identity()
      self.fc2 = torch.nn.Linear(ff_width, 1)
    else:
      self.fc1 = None
  
  def forward(self, x):
    # b n c
    if self.fc1:
      w = torch.nn.functional.silu(self.fc1(x))
      w = self.fc2(self.norm(w))
      w = torch.softmax(w, axis=1)
      return torch.sum(x * w, axis=1)
    else:
      return torch.amax(x, dim=1)