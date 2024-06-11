import torch
import numpy as np
from pytorch_optimizer import DAdaptLion, Prodigy, Lion

class LogRampScheduler():
  def __init__(self, optimizer, min_lr=1e-4, max_lr=1e4, epochs=30):
    super(LogRampScheduler, self).__init__()
    self.lrs = np.geomspace(min_lr, max_lr, epochs)
    self.optimizer = optimizer
    self.momenta = []
    for pg in optimizer.param_groups:
      self.momenta += [self.get_group_momentum(pg)]
      self.set_group_momentum(pg, 0.01)
    self.step_count = 0

  def zero_grad(self):
    pass

  def get_group_momentum(self, pg):
    if "momentum" in pg:
      return pg["momentum"]
    else:
      return pg["betas"][0]

  def set_group_momentum(self, pg, momentum):
    if "momentum" in pg:
      pg["momentum"] = momentum
    else:
      pg["betas"][0] = momentum

  def step(self):
    if self.step_count == 0:
      for pg in self.optimizer.param_groups:
        self.set_group_momentum(pg, 0.5)
    elif self.step_count == 1:
      for pg, momentum in zip(self.optimizer.param_groups, self.momenta):
        self.set_group_momentum(pg, momentum)
    for pg in self.optimizer.param_groups:
      momentum = self.get_group_momentum(pg)
      pg["lr"] = self.lrs[self.step_count] * (1 - momentum)
    self.step_count += 1

  def state_dict(self):
    return {"lrs": self.lrs[:self.step_count]}

class FakeOptimizer():
  def __init__(self, model, verbose=True):
    super(FakeOptimizer, self).__init__()
    if not verbose:
      return
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
  elif "optimizer" in params and params["optimizer"] == "lion":
    optimizer = Lion(model.parameters(),
                     betas=[
                         params["momentum"],
                         params["beta2"],
                     ],
                     weight_decouple=True,
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
    if "optimizer" not in params:
      print("no optimizer specified, defaulting to sgd")
    elif params["optimizer"] != "sgd":
      print(f"unknown optimizer {params['optimizer']}, defaulting to sgd")
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
  elif "schedule" in params and params["schedule"] == "ramp":
    scheduler = LogRampScheduler(
        optimizer,
        min_lr=params["min_lr"],
        max_lr=params["max_lr"],
        epochs=int(params["max_epochs"]))
  else:
    assert "schedule" not in params or params["schedule"] is None, params["schedule"]
    scheduler = FakeOptimizer(model, verbose=False)
  return optimizer, scheduler