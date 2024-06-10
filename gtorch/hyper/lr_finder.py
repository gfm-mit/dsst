import numpy as np
import torch
import itertools
from tqdm import tqdm

import gtorch.models.base
from gtorch.optimize.loss import classify, next_token
from gtorch.optimize.optimizer import get_optimizer

def find_lr(params, model_factory_fn, train_loader=None, pretraining=None):
  assert isinstance(model_factory_fn, gtorch.models.base.Base)
  assert pretraining in [None, "none", "load", "save"]
  if pretraining == "save":
    loss_fn = next_token
    assert isinstance(model_factory_fn, gtorch.models.base.SequenceBase)
    model = model_factory_fn.get_next_token_architecture(**{
        k: params[k]
        for k in "".split()
    })
  else:
    loss_fn = classify
    model = model_factory_fn.get_classifier_architecture(**{
        k: params[k]
        for k in "".split()
    })

  if pretraining == "load":
    network_state_dict = torch.load('./results/model.pth')
    model.load_state_dict(network_state_dict, strict=False)
  else:
    for layer in model.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

  new_params = dict(**params)
  low, high, N = 1e-2, 1e2, 30
  new_params["schedule"] = "exponential"
  new_params["learning_rate"] = low
  new_params["schedule_gamma"] = np.power(high / low, 1.0 / N)
  optimizer, scheduler = get_optimizer(new_params, model)
  lrs = []
  losses = []
  for e, batch in zip(tqdm(range(N)), itertools.cycle(train_loader)):
    losses += [loss_fn(0, model, optimizer, [batch])]
    lrs += [scheduler.get_last_lr()[0]]
    scheduler.step()
    if lrs[-1] > high:
      break
  return lrs, losses