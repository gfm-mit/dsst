import numpy as np
import torch

class OneCat(torch.nn.Module):
  def forward(self, input):
    return torch.cat([torch.zeros([input.shape[0], 1]), input], axis=1)

class PrintCat(torch.nn.Module):
  def forward(self, input):
    print(input.shape)
    return input

class PadCat(torch.nn.Module):
  def __init__(self, width=8):
    super().__init__()
    self.width = width

  def forward(self, input):
    n_h = int(np.ceil(input.shape[2] / self.width)) * self.width
    return torch.nn.functional.pad(input, (0, n_h - input.shape[2]))

class FourierAttention(torch.nn.Module):
    batch_first = False

    def forward(self, query, key, value,
                attn_mask=None,
                key_padding_mask=None,
                is_causal=None,
                need_weights=False):
        return torch.fft.fft(torch.fft.fft(query, axis=0), axis=2).real

class NoopAttention(torch.nn.Module):
    def forward(self, query, key, value,
                attn_mask=None,
                key_padding_mask=None,
                is_causal=None,
                need_weights=False):
        return query