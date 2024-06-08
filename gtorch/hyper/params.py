from pathlib import Path
import time

import numpy as np
import torch

import gtorch.models.base
from gtorch.optimize.loss import classify, next_token
from gtorch.optimize.metrics import binary_classifier_metrics, next_token_metrics
from gtorch.optimize.optimizer import get_optimizer

Path('./results').mkdir(parents=True, exist_ok=True)
def one_training_run(model, optimizer, scheduler, min_epochs, max_epochs, train_loader, loss_fn=None):
  max_loss = 3
  start_time, last_print_time = time.time(), time.time()
  for x in range(max_epochs):
    #train_data = ContrastiveDataset('./TRAIN/', seed=str(np.random.randint(10)))
    #val_data = ContrastiveDataset('./TEST/', seed=str(np.random.randint(10)))
    # TODO: reshuffle data after each epoch, of course
    #train_loader = torch.utils.data.DataLoader(
    #    train_data, batch_size=batch_size_train, shuffle=False, collate_fn=collate_fn_padd)
    #val_loader = torch.utils.data.DataLoader(
    #    val_data, batch_size=batch_size_test, shuffle=False, collate_fn=collate_fn_padd)

    #torch.save(model, './results/model.pth')
    state_dict = dict(**model.state_dict())
    next_loss = loss_fn(x, model, optimizer, train_loader)
    scheduler.step()
    if next_loss > max_loss and x > min_epochs:
      print(f"next_loss too big: {next_loss} > {max_loss}")
      model.load_state_dict(state_dict)
      return dict(roc=0), model
      break
    if np.isnan(next_loss):
      print("next_loss isnan")
      model.load_state_dict(state_dict)
      return dict(roc=0), model
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

def setup_training_run(params, model_factory_fn, train_loader=None, val_loader=None, pretraining=None):
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
    network_state_dict = torch.load('sequence_pretrain.pth')
    model.load_state_dict(network_state_dict, strict=False)
  else:
    for layer in model.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

  optimizer, scheduler = get_optimizer(params, model)

  torch.cuda.empty_cache()
  model = one_training_run(model, optimizer, scheduler,
                           min_epochs=params["min_epochs"],
                           max_epochs=params["max_epochs"],
                           train_loader=train_loader,
                           loss_fn=loss_fn)
  if pretraining == "save":
    resdict = next_token_metrics(model, val_loader)
    return resdict, model
  else:
    resdict = binary_classifier_metrics(model, val_loader)
    return resdict, model