import pathlib
import numpy as np
import torch
from einops.layers.torch import Rearrange
import re
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconConfig, build_alibi_tensor
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import warnings

import models.base
from models.util import PrintfModule
from hashlib import sha1

class FalconWrapper(torch.nn.Module):
  def __init__(self, arch_width, arch_depth, arch_head, arch_dropout, device=None, arch_mask=False):
    super().__init__()
    self.device = device
    self.num_heads = arch_head
    config = FalconConfig(
      hidden_size=arch_width,
      ffn_hidden_size=arch_width,
      num_attention_heads=arch_head,
      num_hidden_layers=arch_depth,
      hidden_dropout=arch_dropout,
      attention_dropout=arch_dropout,
      new_decoder_architecture=True,
      cross_attention_dim=0,
      alibi=True,
    )
    config._attn_implementation = "sdpa"
    self.core = FalconDecoderLayer(config).to(device)
    self.mask_converter = AttentionMaskConverter(is_causal=True)
    self.mask = arch_mask
  
  def forward(self, x):
    batch_size, seq_length, width = x.shape
    mask = self.mask_converter.to_causal_4d(
        batch_size=x.shape[0],
        query_length=x.shape[1],
        key_value_length=x.shape[1],
        dtype=x.dtype,
        device=self.device
    )
    if self.mask:
      alibi_mask = torch.amax(x != 0, dim=2)
      unused_mask = ~torch.einsum("bi,bj->bij", alibi_mask, alibi_mask).to(self.device).unsqueeze(1)
      mask = mask.masked_fill(unused_mask, torch.finfo(mask.dtype).min)
    else:
      alibi_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=self.device)
    alibi = build_alibi_tensor(alibi_mask, self.num_heads, dtype=x.dtype).to(self.device)
    output, = self.core(x, alibi=alibi, attention_mask=mask)
    return output

class CachedEmbeddingWrapper(torch.nn.Module):
  def __init__(self, device, inputs, use_cache=False, **kwargs):
    super().__init__()
    self.project = models.util.SineProjection(inputs, kwargs['arch_width'], axis=-1, scale=1, preserve_zeros=True)
    self.attn = FalconWrapper(device=device, **kwargs)
    self.use_cache = use_cache
    if self.use_cache:
      self.project.requires_grad_(False)
      self.attn.requires_grad_(False)
      if pathlib.Path("./results/cache_data.npy").exists():
        self.cache_data = np.load("./results/cache_data.npy").astype(np.float32)
        self.cache_map = np.load("./results/cache_map.npy", allow_pickle=True).item()
        print(f"loading persistent cache: len(cache)={len(self.cache_map)}")
      else:
        self.cache_data = np.zeros([20_000, 128, 128], dtype=np.float32)
        self.cache_map = {}
    elif pathlib.Path("./results/cache_data.npy").exists():
      pathlib.Path("./results/cache_data.npy").unlink()
      pathlib.Path("./results/cache_map.npy").unlink()
  
  def cache_miss(self, x):
    return self.attn(self.project(x))
  
  # still some bug in this that makes it sometimes fail
  def hash_tensor(self, n_c):
    length = torch.sum(torch.amax(n_c != 0, axis=1))
    return sha1(n_c[:length].cpu().numpy().tobytes()).hexdigest()

  def forward(self, x):
    if not self.use_cache:
      return self.cache_miss(x)
    batch, length, _ = x.shape
    batch_keys = []
    for b in range(batch):
      key = self.hash_tensor(x[b])
      batch_keys += [key]
    if set(batch_keys) - self.cache_map.keys():
      # don't bother subsetting to get new ones
      y = self.cache_miss(x)
      for b, key in enumerate(batch_keys):
        if key not in self.cache_map:
          idx = len(self.cache_map)
          assert idx < self.cache_data.shape[0]
          self.cache_map[key] = idx
          value = y[b].cpu().numpy()
          self.cache_data[idx, :length] = value
      np.save("./results/cache_data.npy", self.cache_data)
      np.save("./results/cache_map.npy", self.cache_map)
      print(f"persisted: len(cache)={len(self.cache_map)}]")
      return y
    #print("cache hit")
    y = np.zeros([batch, length, 128])
    for b, key in enumerate(batch_keys):
      idx = self.cache_map[key]
      y[b] = self.cache_data[idx][:length]
    return torch.from_numpy(y).to(dtype=torch.float32, device=x.device)

class Transformer(models.base.SequenceBase):
  def __init__(self, n_classes=2, n_inputs=12, device='cpu'):
    self.classes = n_classes
    self.inputs = n_inputs
    super().__init__(device=device)
    # dammit, torch ecosystem, this is terrible
    warnings.simplefilter(action='ignore', category=FutureWarning)

  def get_next_token_architecture(self, **kwargs):
    model = torch.nn.Sequential(
        # b n c
        CachedEmbeddingWrapper(device=self.device, inputs=self.inputs, use_cache=False, **kwargs),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(normalized_shape=kwargs['arch_width']),
        torch.nn.Linear(kwargs['arch_width'], self.inputs),
    )
    model = model.to(self.device)
    return model
  
  def translate_state_dict(self, next_token_state_dict):
    classifier_state_dict = {}
    for k, v in next_token_state_dict.items():
      if re.match("0[.](project|attn)[.].*", k):
        print(f"saving param {k}")
        classifier_state_dict[k] = v
      else:
        print(f"not saving param {k}")
    return classifier_state_dict

  def get_classifier_architecture(self, **kwargs):
    # TODO: use_cache should be based on args.disk == "freeze", but I'll be damned if I know where to find that
    model = torch.nn.Sequential(
        # b n c
        CachedEmbeddingWrapper(device=self.device, inputs=self.inputs, use_cache=True, **kwargs),
        torch.nn.SiLU(),
        torch.nn.LayerNorm(normalized_shape=kwargs['arch_width']),
        Rearrange('b n c -> b c n'),
        # TODO: surely this can be improved?
        torch.nn.AdaptiveMaxPool1d(1),
        Rearrange('b c 1 -> b c'),
        torch.nn.Linear(kwargs['arch_width'], self.classes),
        torch.nn.LogSoftmax(dim=-1),
    )
    model = model.to(self.device)
    return model

  def get_next_token_parameters(self):
    return dict(
      scheduler='cosine',
      optimizer='samadam',
      warmup_epochs=8,
      max_epochs=12,
      batch=125,
      learning_rate=3.8e-2
    )

  def get_classifier_parameters(self, **kwargs):
    return dict(
      scheduler='warmup',
      optimizer='samadam',
      weight_decay=0,
      momentum=0.9,
      conditioning_smoother=0.999,
      warmup_epochs=5,
      max_epochs=20,
      learning_rate=2e-2,

      arch_depth=1,
      arch_width=128,
      arch_dropout=0.05,
      arch_head=4,
      arch_mask=False
    )