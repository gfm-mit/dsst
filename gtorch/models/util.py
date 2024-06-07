#@title def get_jank_model(Linear):
import numpy as np
import pandas as pd
import torch

class OneCat(torch.nn.Module):
  def forward(self, input):
    return torch.cat([torch.zeros([input.shape[0], 1]), input], axis=1)

class PrintCat(torch.nn.Module):
  def forward(self, input):
    print(input.shape)
    return input