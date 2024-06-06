from pathlib import Path

import numpy as np
import torch

import gtorch.models.base
from gtorch.optimize.metrics import metrics
from gtorch.optimize.optimize import get_optimizer, optimize

Path('./results').mkdir(parents=True, exist_ok=True)
def one_training_run(model, optimizer, scheduler, min_epochs, max_epochs, train_loader, val_loader):
  max_loss = 3
  resdict = dict(loss=np.nan, accuracy=np.nan)
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
    next_loss = optimize(x, model, optimizer, train_loader)
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
    if x % 10 == 0 or x == max_epochs - 1:
      print('Train Epoch: {} \tPerplexity: {:.2f}'.format(x, np.exp(next_loss)))
  resdict = metrics(model, val_loader)
  return resdict, model

def setup_training_run(params, model_factory_fn, pretrained=False, train_loader=None, val_loader=None):
  assert isinstance(model_factory_fn, gtorch.models.base.Base)
  model = model_factory_fn.get_architecture(**{
      k: params[k]
      for k in "hidden_width".split()
  })

  if pretrained:
    network_state_dict = torch.load('sequence_pretrain.pth')
    model.load_state_dict(network_state_dict, strict=False)
  else:
    for layer in model.children():
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

  optimizer, scheduler = get_optimizer(params, model)

  torch.cuda.empty_cache()
  return one_training_run(model, optimizer, scheduler,
                        min_epochs=params["min_epochs"],
                        max_epochs=params["max_epochs"],
                        train_loader=train_loader,
                        val_loader=val_loader)