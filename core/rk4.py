import numpy as np
import torch
import torch.optim
import pytorch_optimizer

import pandas as pd

class RK4(torch.optim.Optimizer, pytorch_optimizer.base.optimizer.BaseOptimizer):
    def __init__(self, params, named, lr=1e-2):
        super().__init__(params, dict(lr=lr))
        for k, v in named:
          self.state[v]['name'] = k

    def __str__(self) -> str:
        return 'RK4'

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, get_grads=None):
      assert get_grads is not None

      with torch.enable_grad():
          loss = get_grads()

      with torch.no_grad():
        for group in self.param_groups:
          for p in group['params']:
            if p.grad is None:
              continue
            self.state[p]['x0'] = p.clone()

            self.state[p]['dx'] = 1 / 6 * p.grad * group['lr']
            p.add_(-0.5 * p.grad * group['lr'])

      with torch.enable_grad():
          get_grads()

      with torch.no_grad():
        for group in self.param_groups:
          for p in group['params']:
            if p.grad is None:
              continue

            self.state[p]['dx'].add_(1 / 3 * p.grad * group['lr'])
            p.data = self.state[p]['x0'] - 0.5 * p.grad * group['lr']

      with torch.enable_grad():
          get_grads()

      with torch.no_grad():
        for group in self.param_groups:
          for p in group['params']:
            if p.grad is None:
              continue

            self.state[p]['dx'].add_(1 / 3 * p.grad * group['lr'])
            p.data = self.state[p]['x0'] - p.grad * group['lr']

      with torch.enable_grad():
          get_grads()

      with torch.no_grad():
        for group in self.param_groups:
          for p in group['params']:
            if p.grad is None:
              continue

            self.state[p]['dx'] += 1 / 6 * p.grad * group['lr']
            p.data = self.state[p]['x0'] - self.state[p]['dx']
      return loss