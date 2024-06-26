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
def one_training_run(model, optimizer, scheduler, warmup_epochs, max_epochs, train_loader, task=None, tqdm_prefix=None, loss_history_loader=None, offset=1):
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
    loss_upper_bound = 10 * train_loss
    torch.cuda.empty_cache()
    progress.set_postfix_str(" " + loss_description)
    if loss_history_loader is not None:
      val_loss = core.metrics.evaluate(model, loss_history_loader, task, offset=offset)
      epoch_loss_history += [val_loss]
    else:
      epoch_loss_history += [train_loss]
  return model, epoch_loss_history

def setup_model(params, model_factory_fn, task="classify", disk="none"):
  assert isinstance(model_factory_fn, models.base.Base)
  assert task in "classify classify_patient next_token".split()
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

def freeze_loaded_params(missing_keys, unexpected_keys, model, freeze=True):
    sorted_keys = defaultdict(list)
    loaded_params = dict(model.named_parameters())
    for k in missing_keys:
      kk, v = k.split(".", 1)
      sorted_keys[kk] += [v]
      if k in loaded_params:
        del loaded_params[k]
    missing_keys = dict(sorted_keys)
    sorted_keys = defaultdict(list)
    for k in unexpected_keys:
      kk, v = k.split(".", 1)
      sorted_keys[kk] += [v]
    unexpected_keys = dict(sorted_keys)
    print(f"loading.{missing_keys=}\nloading.{unexpected_keys=}\nfreezing: {list(loaded_params.keys())=}")
    if freeze:
      for v in loaded_params.values():
        v.requires_grad = False

def setup_training_run(params, model_factory_fn, train_loader=None, val_loader=None,
                       task="classify", disk="none",
                       tqdm_prefix=None, history='none', offset=1):
  model = setup_model(params, model_factory_fn, task, disk)
  optimizer, scheduler = get_optimizer_and_scheduler(params, model)

  torch.cuda.empty_cache()
  model, epoch_loss_history = one_training_run(model, optimizer, scheduler,
                                   warmup_epochs=params["warmup_epochs"],
                                   max_epochs=params["max_epochs"],
                                   train_loader=train_loader,
                                   task=task,
                                   tqdm_prefix=tqdm_prefix,
                                   loss_history_loader=val_loader if history == "val" else None, offset=offset)
  if disk == "save":
    state_dict = model.state_dict()
    if task == "next_token":
      state_dict = model_factory_fn.translate_state_dict(state_dict)
    torch.save(state_dict, './results/model.pth')
  metric = core.metrics.evaluate(model, val_loader, task, offset=offset)
  return metric, epoch_loss_history, model