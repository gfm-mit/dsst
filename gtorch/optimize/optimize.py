from pathlib import Path

import torch
from pytorch_optimizer import DAdaptLion, Prodigy

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
  if epoch % 10 == 0:
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
  return loss.item()

class FakeOptimizer():
  def __init__(self, model):
    super(FakeOptimizer, self).__init__()
    layer_lookup = {
        k: str(v)
        for k, v in model.named_modules()
        if k
    }
    parameters = [
        (list(map(str, v.shape)), *k.split(".", maxsplit=1))
        for k, v in model.named_parameters()
    ]
    for s, l, p in parameters:
      print("{:>10}: {:10}<-{}".format(
          ", ".join(s[::-1]), p, layer_lookup.get(l, l)
      ))

  def zero_grad(self):
    pass

  def step(self):
    pass

  def state_dict(self):
    return {}


def get_optimizer(params, model):
  if "optimizer" in params and params["optimizer"] == "adam":
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params["learning_rate"],
                                  betas=[
                                      params["momentum"],
                                      params["beta2"],
                                  ],
                                  weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "dadaptlion":
    optimizer = DAdaptLion(model.parameters(),
                           betas=[
                               params["momentum"],
                               params["beta2"],
                           ],
                           weight_decouple=True,
                           weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "prodigy":
    optimizer = Prodigy(model.parameters(),
                        betas=[
                            params["momentum"],
                            params["beta2"],
                        ],
                        weight_decouple=True,
                        weight_decay=params["weight_decay"])
  else:
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params["learning_rate"],
                                momentum=params["momentum"],
                                weight_decay=params["weight_decay"])
  if "schedule" in params and params["schedule"] == "onecycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params["learning_rate"],
        steps_per_epoch=1,
        pct_start=params["pct_start"],
        epochs=int(params["max_epochs"]))
  else:
    scheduler = FakeOptimizer(model)
  return optimizer, scheduler