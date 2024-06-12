from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

import gtorch.models.base
import gtorch.optimize
import gtorch.optimize.loss
import gtorch.optimize.metrics
from gtorch.optimize.optimizer import get_optimizer

Path('./results').mkdir(parents=True, exist_ok=True)
def one_training_run(model, optimizer, scheduler, min_epochs, max_epochs, train_loader, task=None, tqdm_prefix=None, loss_history_loader=None):
  loss_upper_bound = 8 # only for the first step
  start_time = time.time()
  if tqdm_prefix is None:
    tqdm_prefix = ""
  progress = tqdm(range(max_epochs), desc=tqdm_prefix)
  epoch_loss_history = []
  for epoch in progress:
    state_dict = dict(**model.state_dict())
    train_loss, loss_description = gtorch.optimize.loss.get_task_loss(epoch, model, optimizer, train_loader, task=task)
    scheduler.step()
    if train_loss > loss_upper_bound and epoch > min_epochs:
      print(f"next_loss too big: {train_loss} > {loss_upper_bound}")
      model.load_state_dict(state_dict)
      return model, epoch_loss_history
    if np.isnan(train_loss):
      print("next_loss isnan")
      model.load_state_dict(state_dict)
      return model, epoch_loss_history
    loss_upper_bound = 1.5 * train_loss
    torch.cuda.empty_cache()
    progress.set_description(tqdm_prefix + loss_description)
    if loss_history_loader is not None:
      val_loss = gtorch.optimize.metrics.evaluate(model, loss_history_loader, task)
      epoch_loss_history += [val_loss]
    else:
      epoch_loss_history += [train_loss]
  print("Train time per epoch", np.round((time.time() - start_time) / (epoch + 1), 2))
  return model, epoch_loss_history

def setup_model(params, model_factory_fn, task="classify", disk="none"):
  assert isinstance(model_factory_fn, gtorch.models.base.Base)
  assert task in "classify classify_patient next_token".split()
  assert disk in "none load save".split()
  if task == "next_token":
    assert isinstance(model_factory_fn, gtorch.models.base.SequenceBase)
    model = model_factory_fn.get_next_token_architecture(**{
        k: params[k]
        for k in "".split()
    })
  else:
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
  return model

def setup_training_run(params, model_factory_fn, train_loader=None, val_loader=None,
                       task="classify", disk="none",
                       tqdm_prefix=None, history='none'):
  model = setup_model(params, model_factory_fn, task, disk)
  optimizer, scheduler = get_optimizer(params, model)

  torch.cuda.empty_cache()
  model, epoch_loss_history = one_training_run(model, optimizer, scheduler,
                                   min_epochs=params["min_epochs"],
                                   max_epochs=params["max_epochs"],
                                   train_loader=train_loader,
                                   task=task,
                                   tqdm_prefix=tqdm_prefix,
                                   loss_history_loader=val_loader if history == "val" else None)
  if disk == "save":
    torch.save(model.state_dict(), './results/model.pth')
  metric = gtorch.optimize.metrics.evaluate(model, val_loader, task)
  return metric, epoch_loss_history, model