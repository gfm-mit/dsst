import torch

def balance_class_weights(target):
    w = torch.nn.functional.one_hot(target).float().mean(axis=0)
    w = 1 / w
    w = torch.clip(w, 0, 10)
    w = w / w.sum()
    return w

def classify(epoch, model, optimizer, train_loader):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
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
    optimizer.step()
  assert loader_has_batches
  return loss.item()

def next_token(epoch, model, optimizer, train_loader):
  DEVICE = next(model.parameters()).device
  model.train()
  loader_has_batches = False
  for data, target in train_loader:
    loader_has_batches = True
    optimizer.zero_grad()
    output = model(data.to(DEVICE))
    loss = torch.nn.functional.mse_loss(output[:, :-1, :], data[:, 1:, :].to(DEVICE))
    loss.backward()
    optimizer.step()
  assert loader_has_batches
  return loss.item()