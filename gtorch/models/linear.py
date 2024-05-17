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

class OneCat(torch.nn.Module):
  def forward(self, input):
    return torch.cat([torch.zeros([input.shape[0], 1]), input], axis=1)

class PrintCat(torch.nn.Module):
  def forward(self, input):
    print(input[:, :, 0].T)
    return input

def get_model(hidden_width=512, device='cuda', classes=2):
  assert classes == 2, classes
  device = 'cpu' #TODO: jank!!!!
  model = torch.nn.Sequential(
      #NegCat(),
      #PrintCat(),
      Rearrange('b c 1 -> b c'),
      # torch.nn.Linear(12, classes),
      # torch.nn.Softmax(dim=-1),
      # why is 2 class prediction worse!?!
      torch.nn.Linear(6, 1),
      OneCat(),
      torch.nn.LogSoftmax(dim=-1),
  )
  model = model.to(device)
  base_params = dict(
    weight_decay=0,
    momentum=0.,
    beta2=0.,
    pct_start=0.1,

    max_epochs=1,
    min_epochs=1,

    learning_rate=1e-2, # unclear if this is right
    hidden_width=2,
    tune=dict(
      #learning_rate=np.geomspace(1e-5, 1e+0, 35),
      #weight_decay=np.geomspace(1e-5, 1e+3, 35),
    ),
  )
  model.state_dict()['1.bias'][0] = -2.65
  #artist = plt.scatter(np.arange(7), [-2.64393084, 1.49190833, -0.79180225, 0.57461165, 0.18255898, 0.42163847, -0.13575019], color='lightgray', zorder=-10, s=200)
  return model, base_params