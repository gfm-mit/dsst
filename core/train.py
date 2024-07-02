from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import models.base
import core.optimizer
import core.loss
import core.metrics
from core.optimizer import get_optimizer_and_scheduler

Path('./results').mkdir(parents=True, exist_ok=True)
MAX_ENTROPY = 80
MAX_RMSE = 80
MAX_JUMP = 10
def one_training_run(model, optimizer, scheduler, warmup_epochs, max_epochs, train_loader, task=None, tqdm_prefix=None, early_stopping_loader=None, offset=1):
  loss_upper_bound = 8 # only for the first step
  progress = tqdm(range(max_epochs), desc=tqdm_prefix or "")
  epoch_loss_history = []
  for epoch in progress:
    state_dict = dict(**model.state_dict())
    train_loss, loss_description = core.loss.get_task_loss(model, optimizer, train_loader, task=task, offset=offset)
    scheduler.step()
    if train_loss > loss_upper_bound and epoch > warmup_epochs:
      print(f"next_loss too big: {train_loss} > {loss_upper_bound}")
      model.load_state_dict(state_dict)
      return model, epoch_loss_history
    if np.isnan(train_loss):
      print("next_loss isnan")
      model.load_state_dict(state_dict)
      return model, epoch_loss_history
    if task in "classify classify_patient classify_section".split():
      loss_upper_bound = min(MAX_JUMP * train_loss, MAX_ENTROPY)
    else:
      loss_upper_bound = min(MAX_JUMP * train_loss, MAX_RMSE)
    torch.cuda.empty_cache()
    progress.set_postfix_str(" " + loss_description)
    if early_stopping_loader is None:
      epoch_loss_history += [train_loss]
    else:
      val_loss = core.metrics.evaluate(model, early_stopping_loader, task, offset=offset)
      epoch_loss_history += [val_loss]
      # TODO: flag to disable saving early, but only when needed
      if core.metrics.best_so_far(epoch_loss_history, task):
        torch.save(model.state_dict(), './results/early.pth')
  return model, epoch_loss_history

def setup_model(params, model_factory_fn, task="classify", disk="none"):
  assert isinstance(model_factory_fn, models.base.Base)
  assert task in "classify classify_patient classify_section next_token".split()
  assert disk in "none load save freeze".split()
  overlay_params = {
    k: params[k]
    for k in params.keys()
    if k.startswith("arch_")
    # TODO: namespace by having sub dictionaries, instead
  }
  if task == "next_token":
    assert isinstance(model_factory_fn, models.base.SequenceBase)
    model = model_factory_fn.get_next_token_architecture(**overlay_params)
  else:
    model = model_factory_fn.get_classifier_architecture(**overlay_params)

  if disk in "load freeze".split():
    network_state_dict = torch.load('./results/model.pth')
    missing_keys, unexpected_keys = model.load_state_dict(network_state_dict, strict=False)
    # TODO: set a flag for whether the parameters are frozen
    freeze_loaded_params(missing_keys, unexpected_keys, model, freeze=disk == "freeze")
  else:
    for layer in model.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  # TODO: try this again when compiling to CUDA
  # DEVICE = next(model.parameters()).device
  # if DEVICE == "cuda":
  #   model = torch.compile(model)
  return model

def nest_param_list(flat_list):
  result = {}
  for key in flat_list:
    parts = [y for x in key.split(".") for y in x.split("_")]
    d = result
    for part in parts[:-1]:
      if part not in d:
        d[part] = {}
      d = d[part]
    d[parts[-1]] = None
  for key in flat_list:
    parts = [y for x in key.split(".") for y in x.split("_")]
    d = result
    for part in parts[:-2]:
      if part not in d:
        d[part] = {}
      d = d[part]
    part = parts[-2]
    if isinstance(d[part], dict) and all([x is None for x in d[part].values()]):
      d[part] = set(d[part].keys())
  return result

def freeze_loaded_params(missing_keys, unexpected_keys, model, freeze=True):
    loaded_params = dict(model.named_parameters())
    for k in missing_keys:
      if k in loaded_params:
        del loaded_params[k]
    missing_keys = nest_param_list(missing_keys)
    unexpected_keys = nest_param_list(unexpected_keys)
    frozen_keys = nest_param_list(loaded_params.keys())
    print(f"loading.{missing_keys=}\nloading.{unexpected_keys=}\nloading.{frozen_keys=}")
    if freeze:
      for v in loaded_params.values():
        v.requires_grad = False

def setup_training_run(params, model_factory_fn, train_loader=None, val_loader=None,
                       task="classify", disk="none",
                       tqdm_prefix=None, use_loss_history=False, offset=1):
  model = setup_model(params, model_factory_fn, task, disk)
  optimizer, scheduler = get_optimizer_and_scheduler(params, model)

  torch.cuda.empty_cache()
  model, epoch_loss_history = one_training_run(model, optimizer, scheduler,
                                   warmup_epochs=int(params["warmup_epochs"]),
                                   max_epochs=int(params["max_epochs"]),
                                   train_loader=train_loader,
                                   task=task,
                                   tqdm_prefix=tqdm_prefix,
                                   early_stopping_loader=None if use_loss_history else val_loader,
                                   offset=offset)
  if not use_loss_history:
    network_state_dict = torch.load('./results/early.pth')
    model.load_state_dict(network_state_dict, strict=True)
  if disk == "save":
    state_dict = model.state_dict()
    if task in "next_token classify_section".split():
      state_dict = model_factory_fn.translate_state_dict(state_dict)
    torch.save(state_dict, './results/model.pth')
  metric = core.metrics.evaluate(model, val_loader, task, offset=offset)
  return metric, epoch_loss_history, model