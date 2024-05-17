#@title def get_jank_model(Linear):
import numpy as np
import pandas as pd
import torch
import shutil
from pathlib import Path
import re
import matplotlib.pyplot as plt
import scipy
from tqdm.notebook import tqdm
import einops
from einops.layers.torch import Rearrange
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class NegCat(torch.nn.Module):
  def forward(self, input):
    #return torch.cat([input, -input, torch.ones([input.shape[0], 1, 1])], axis=1)
    return torch.cat([input, -input], axis=1)
class ProbCat(torch.nn.Module):
  def forward(self, input):
    return torch.cat([input, 1-input], axis=1)

def get_model(hidden_width=512, device='cuda', classes=2):
  assert classes == 2 # oof
  device = 'cpu' #TODO: jank!!!!
  model = torch.nn.Sequential(
      NegCat(),
      Rearrange('b c 1 -> b c'),
      # torch.nn.Linear(12, classes),
      # torch.nn.Softmax(dim=-1),
      # why is 2 class prediction worse!?!
      torch.nn.Linear(12, 1),
      torch.nn.Softmax(dim=-2),
      ProbCat(),
  )
  model = model.to(device)
  base_params = dict(
    weight_decay=3e1,
    momentum=0.,
    beta2=0.,
    pct_start=0.1,

    max_epochs=5,
    min_epochs=2,

    learning_rate=1e-2, # unclear if this is right
    hidden_width=2,
  )
  return model, base_params