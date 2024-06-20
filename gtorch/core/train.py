from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import models.base
import gtorch.core.optimizer
import gtorch.core.loss
import gtorch.core.metrics
from gtorch.core.optimizer import get_optimizer_and_scheduler

Path('./results').mkdir(parents=True, exist_ok=True)
def one_training_run(model, optimizer, scheduler, min_epochs, max_epochs, train_loader, task=None, tqdm_prefix=None, loss_history_loader=None):
  loss_upper_bound = 8 # only for the first step
  progress = tqdm(range(max_epochs), desc=tqdm_prefix or "")
  epoch_loss_history = []
  for epoch in progress:
    state_dict = dict(**model.state_dict())
    train_loss, loss_description = gtorch.core.loss.get_task_loss(model, optimizer, train_loader, task=task)
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
    progress.set_postfix_str(" " + loss_description)
    if loss_history_loader is not None:
      val_loss = gtorch.core.metrics.evaluate(model, loss_history_loader, task)
      epoch_loss_history += [val_loss]
    else:
      epoch_loss_history += [train_loss]
  return model, epoch_loss_history

def setup_model(params, model_factory_fn, task="classify", disk="none"):
  assert isinstance(model_factory_fn, models.base.Base)
  assert task in "classify classify_patient next_token".split()
  assert disk in "none load save".split()
  overlay_params = {
    k: params[k]
    for k in "".split()
    # TODO: figure out where this needs to be controlled
  }
  if task == "next_token":
    assert isinstance(model_factory_fn, models.base.SequenceBase)
    model = model_factory_fn.get_next_token_architecture(**overlay_params)
  else:
    model = model_factory_fn.get_classifier_architecture(**overlay_params)

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
  optimizer, scheduler = get_optimizer_and_scheduler(params, model)

  torch.cuda.empty_cache()
  model, epoch_loss_history = one_training_run(model, optimizer, scheduler,
                                   min_epochs=params.get("min_epochs", params.get("warmup_steps", 0)),
                                   max_epochs=params["max_epochs"],
                                   train_loader=train_loader,
                                   task=task,
                                   tqdm_prefix=tqdm_prefix,
                                   loss_history_loader=val_loader if history == "val" else None)
  if disk == "save":
    torch.save(model.state_dict(), './results/model.pth')
  metric = gtorch.core.metrics.evaluate(model, val_loader, task)
  return metric, epoch_loss_history, model