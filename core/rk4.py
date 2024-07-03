import torch
import torch.optim
import pytorch_optimizer

class RK4(torch.optim.Optimizer, pytorch_optimizer.base.optimizer.BaseOptimizer):
    def __init__(self, params, lr=1e-2):
        super().__init__(params)
        self.learning_rate = lr

    def __str__(self) -> str:
        return 'RK4'

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, get_grads=None):
        assert get_grads is not None

        with torch.enable_grad():
          get_grads()

        for group in self.param_groups:
          for p in group['params']:
            if p.grad is None:
              continue
            self.state[p]['x0'] = p.clone()
            self.state[p]['k'] = p.grad.clone()
            p.add_(p.grad / 2)

        with torch.enable_grad():
          get_grads()

        for group in self.param_groups:
          for p in group['params']:
            if p.grad is None:
              continue

            self.state[p]['k'] += 2 * p.grad
            p.data = self.state[p]['x0']
            p.add_(p.grad / 2)

        with torch.enable_grad():
          get_grads()

        for group in self.param_groups:
          for p in group['params']:
            if p.grad is None:
              continue

            self.state[p]['k'] += 2 * p.grad
            p.data = self.state[p]['x0']
            p.add_(p.grad)

        with torch.enable_grad():
          get_grads()

        for group in self.param_groups:
          for p in group['params']:
            if p.grad is None:
              continue

            self.state[p]['k'] += p.grad
            p.data = self.state[p]['x0'] + self.state[p]['k'] / 6 * self.lr