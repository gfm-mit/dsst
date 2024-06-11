import numpy as np
import itertools
from tqdm import tqdm

import gtorch.models.base
import gtorch.hyper.params
from gtorch.optimize.optimizer import get_optimizer

def find_lr(params, model_factory_fn, train_loader=None, task="classify", disk="none"):
  model, loss_fn = gtorch.hyper.params.setup_model(params, model_factory_fn, task, disk)

  new_params = dict(**params)
  low, high, N = 3e-2, 3e0, 20
  lrs = []
  losses = []
  for e, batch in zip(tqdm(range(N)), itertools.cycle(train_loader)):
    lr = low * np.power(high / low, float(e) / N)
    new_params["learning_rate"] = lr
    lrs += [lr]
    optimizer, scheduler = get_optimizer(new_params, model)
    losses += [loss_fn(0, model, optimizer, [batch])]
  return lrs, losses