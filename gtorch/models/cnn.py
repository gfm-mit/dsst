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
import torchvision
from einops.layers.torch import Rearrange
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_model(hidden_width=32, device='cpu', classes=2):
  width2 = hidden_width // 2
  model = torch.nn.Sequential(
      torch.nn.Conv2d(
          in_channels=1,
          out_channels=width2,
          kernel_size=3,
          stride=1,
          padding=1,
          bias=False,
      ),
      torch.nn.ReLU6(),
      torch.nn.BatchNorm2d(width2), 
      torch.nn.MaxPool2d(2, 2),

      # efficientnet block
      torch.nn.Conv2d(
          in_channels=width2,
          out_channels=hidden_width,
          kernel_size=1,
          stride=1,
          padding=0,
          bias=False,
      ),
      torch.nn.ReLU6(),
      torch.nn.BatchNorm2d(hidden_width), 
      torch.nn.Conv2d(
          in_channels=hidden_width,
          groups=hidden_width,
          out_channels=hidden_width,
          kernel_size=3,
          stride=1,
          padding=1,
          bias=False,
      ),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(hidden_width), 
      torchvision.ops.SqueezeExcitation(hidden_width, hidden_width),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(hidden_width), 
      torch.nn.Conv2d(
          in_channels=hidden_width,
          out_channels=width2,
          kernel_size=1,
          stride=1,
          padding=0,
          bias=False,
      ),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(width2), 

      # some stuff
      torch.nn.AdaptiveMaxPool2d(1),
      Rearrange('b c 1 1 -> b c'),
      torch.nn.Linear(width2, 2),
      torch.nn.Softmax(dim=-1),
  )
  model = model.to(device)
  base_params = dict(
    weight_decay=1e-4,
    momentum=0.,
    beta2=0.,
    pct_start=0.1,

    max_epochs=5,
    min_epochs=2,

    learning_rate=1e-4,
    hidden_width=8,
  )
  return model, base_params