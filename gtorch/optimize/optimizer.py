import torch
import numpy as np
from pytorch_optimizer import DAdaptLion, Prodigy

class LogRampScheduler():
  def __init__(self, optimizer, min_lr=1e-4, max_lr=1e4, epochs=30):
    super(LogRampScheduler, self).__init__()
    self.lrs = np.geomspace(min_lr, max_lr, epochs)
    self.optimizer = optimizer
    self.momenta = []
    for pg in optimizer.param_groups:
      self.momenta += [pg["momentum"]]
      pg["momentum"] = 0.01
    self.step_count = 0

  def zero_grad(self):
    pass

  def step(self):
    if self.step_count == 0:
      for pg in self.optimizer.param_groups:
        pg["momentum"] = 0.5
    elif self.step_count == 1:
      for pg, momentum in zip(self.optimizer.param_groups, self.momenta):
        pg["momentum"] = momentum
    for pg in self.optimizer.param_groups:
      pg["lr"] = self.lrs[self.step_count] * (1 - pg["momentum"])
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