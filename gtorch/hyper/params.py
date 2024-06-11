from pathlib import Path
import time

import numpy as np
import torch

import gtorch.models.base
import gtorch.optimize
from gtorch.optimize.loss import classify, next_token
import gtorch.optimize.metrics
from gtorch.optimize.optimizer import get_optimizer

Path('./results').mkdir(parents=True, exist_ok=True)
def one_training_run(model, optimizer, scheduler, min_epochs, max_epochs, train_loader, loss_fn=None):
  max_loss = 3
  start_time, last_print_time = time.time(), time.time()
  for x in range(max_epochs):
    state_dict = dict(**model.state_dict())
    next_loss = loss_fn(x, model, optimizer, train_loader)
    scheduler.step()
    if next_loss > max_loss and x > min_epochs:
      print(f"next_loss too big: {next_loss} > {max_loss}")
      model.load_state_dict(state_dict)
      return model
      break
    if np.isnan(next_loss):
      print("next_loss isnan")
      model.load_state_dict(state_dict)
      return model
      break
    max_loss = 1.5 * next_loss
    torch.cuda.empty_cache()
    if x % 10 == 0 or x == max_epochs - 1 or time.time() - last_print_time > 15:
      if loss_fn == classify:
        print('Train Epoch: {} \tLast Batch Perplexity: {:.2f}'.format(x, np.exp(next_loss)))
      else:
        print('Train Epoch: {} \tLast Batch MSE: {:.2f}'.format(x, next_loss))
      last_print_time = time.time()
  print("Train time per epoch", np.round((time.time() - start_time) / (x + 1), 2))
  return model

def setup_model(params, model_factory_fn, task="classify", disk="none"):
  assert isinstance(model_factory_fn, gtorch.models.base.Base)
  assert task in "classify next_token".split()
  assert disk in "none load save".split()
  if task == "next_token":
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

  if disk == "load":
    network_state_dict = torch.load('./results/model.pth')
    model.load_state_dict(network_state_dict, strict=False)
  else:
    for layer in model.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  return model, loss_fn

def setup_training_run(params, model_factory_fn, train_loader=None, val_loader=None, task="classify", disk="none"):
  model, loss_fn = setup_model(params, model_factory_fn, task, disk)
  optimizer, scheduler = get_optimizer(params, model)

  torch.cuda.empty_cache()
  model = one_training_run(model, optimizer, scheduler,
                           min_epochs=params["min_epochs"],
                           max_epochs=params["max_epochs"],
                           train_loader=train_loader,
                           loss_fn=loss_fn)
  if disk == "save":
    torch.save(model.state_dict(), './results/model.pth')
  return gtorch.optimize.metrics.evaluate(model, val_loader, task)