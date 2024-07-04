import itertools
import numpy as np
from tqdm import trange

import models.base
import core.train
import optim.loss
import optim.scheduler
from optim.optimizer import get_optimizer_and_scheduler


def find_lr(params, model_factory_fn, train_loader=None, task="classify", disk="none", tqdm_desc=""):
  model = core.train.setup_model(params, model_factory_fn, task, disk)

  new_params = dict(**params)
  losses = []
  conds = []
  assert new_params['optimizer'] != 'prodigy'  # prodigy LR is weird
  optimizer, scheduler = get_optimizer_and_scheduler(new_params, model)
  assert isinstance(scheduler, optim.scheduler.LogRampScheduler)
  last_grads = None
  progress = trange(new_params["max_epochs"], desc=tqdm_desc)
  for e, batch in zip(progress, itertools.cycle(train_loader)):
    losses += [optim.loss.get_task_loss(model, optimizer, [batch], task)[0]]
    scheduler.step()
    grads = np.concatenate([t.detach().cpu().numpy().flatten() for t in model.parameters()])
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

    if np.argmin(losses) == len(losses) - 1:
      progress.set_postfix_str(f"loss: {losses[-1]:.2E} @ step {e}")
    if e > 0 and losses[-1] > losses[0] * 2:
      break
  lrs = scheduler.state_dict()["lrs"]
  return lrs, losses, conds