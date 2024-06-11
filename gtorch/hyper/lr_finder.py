import itertools
import numpy as np
import torch
from tqdm import tqdm

import gtorch.models.base
import gtorch.hyper.params
from gtorch.optimize.optimizer import get_optimizer


def find_lr(params, model_factory_fn, train_loader=None, task="classify", disk="none"):
  model, loss_fn = gtorch.hyper.params.setup_model(params, model_factory_fn, task, disk)

  new_params = dict(**params)
  losses = []
  conds = []
  optimizer, scheduler = get_optimizer(new_params, model)
  last_grads = None
  for e, batch in zip(tqdm(range(new_params["max_epochs"])), itertools.cycle(train_loader)):
    losses += [loss_fn(0, model, optimizer, [batch])]
    scheduler.step()
    grads = np.concatenate([t.detach().numpy().flatten() for t in model.parameters()])
    if last_grads is not None:
      plus = grads + last_grads
      plus /= np.linalg.norm(plus)
      minus = grads - last_grads
      minus /= np.linalg.norm(minus)
      plus2 = np.sum(plus * grads) - np.sum(plus * last_grads)
      minus2 = np.sum(minus * grads) - np.sum(minus * last_grads)
      cond = np.abs(minus2 / plus2)
      conds += [cond]
    else:
      conds = [np.nan]
    last_grads = grads

    if disk == "save" and losses and np.argmin(losses) == len(losses) - 1:
      torch.save(model.state_dict(), './results/model.pth')

    if e > 0 and losses[-1] > losses[0] * 2:
      break
  lrs = scheduler.state_dict()["lrs"]
  return lrs, losses, conds