from pathlib import Path

import torch

Path('./results/').mkdir(parents=True, exist_ok=True)

def balance_class_weights(target):
    w = torch.nn.functional.one_hot(target).float().mean(axis=0)
    w = 1 / w
    w = torch.clip(w, 0, 10)
    w = w / w.sum()
    return w

def optimize(epoch, model, optimizer, train_loader):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  #gradients = {}
  #
  #def hook(module, grad_input, grad_output):
  #    print(type(module), grad_input[0].shape, grad_output[0].shape)
  #    gradients[module] = torch.cat([
  #       grad_input[0],
  #       #torch.ones(grad_input[0].shape[0], 1).to(DEVICE),
  #       grad_output[0],
  #    ], axis=1)
  #    gradients[str(module) + "_in"] = grad_input[0].shape
  #    gradients[str(module) + "_out"] = grad_output[0].shape
  #
  #for layer in model.children():
  #  layer.register_backward_hook(hook)
  for data, target in train_loader:
    loader_has_batches = True
    optimizer.zero_grad()
    output = model(data.to(DEVICE))
    #class_weights = balance_class_weights(target)
    #loss = torch.nn.functional.nll_loss(
    #    output, target.to(DEVICE), weight=class_weights.to(DEVICE))
    # TODO: if you change this, stop reporting loss as a perplexity
    loss = torch.nn.functional.nll_loss(output, target[:, 0].to(DEVICE))
    loss.backward()
    #print("target", target[0])
    #for k, v in gradients.items():
    #  print(k, v)
    #print("output.grad", output.grad)
    #for k, v in model.named_parameters():
    #  print(k, v.grad)
    optimizer.step()
  assert loader_has_batches
  torch.save(model.state_dict(), './results/model.pth')
  return loss.item()