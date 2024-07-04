import pytorch_optimizer.lr_scheduler
import torch
import pytorch_optimizer
import numpy as np

class LogRampScheduler():
  def __init__(self, optimizer, min_lr=None, max_lr=None, epochs=None):
    super(LogRampScheduler, self).__init__()
    self.lrs = np.geomspace(min_lr, max_lr, epochs)
    self.optimizer = optimizer
    self.momenta = []
    for pg in optimizer.param_groups:
      pg["lr"] = self.lrs[0]
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


def get_scheduler(params, model, optimizer):
  key = "scheduler"
  assert "schedule" not in params
  if key in params and params[key] == "onecycle":
    # damned thing uses cosine decay, anyway
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params["learning_rate"],
        steps_per_epoch=1,
        pct_start=float(params["warmup_epochs"]) / params["max_epochs"],
        epochs=int(params["max_epochs"]))
  elif key in params and params[key] == "cosine":
    scheduler = pytorch_optimizer.lr_scheduler.linear_warmup.CosineScheduler(
        optimizer,
        max_lr=params["learning_rate"],
        warmup_steps=params["warmup_epochs"],
        t_max=int(params["max_epochs"]))
  elif key in params and params[key] == "warmup":
    scheduler = pytorch_optimizer.lr_scheduler.linear_warmup.LinearScheduler(
        optimizer,
        init_lr=params["init_lr"],
        max_lr=params["learning_rate"],
        min_lr=params["learning_rate"], # this is final_lr, annoying hack to get constant LR after warmup
        warmup_steps=params["warmup_epochs"],
        t_max=int(params["max_epochs"]))
  elif key in params and params[key] == "ramp":
    scheduler = LogRampScheduler(
        optimizer,
        min_lr=params.get("min_lr", None),
        max_lr=params.get("max_lr", None),
        epochs=int(params["max_epochs"]))
  else:
    assert key not in params or params[key] is None or params[key] == "none", params[key]
    assert "warmup_epochs" not in params or params["warmup_epochs"] == 0
    scheduler = FakeOptimizer(model, verbose=False)
  return optimizer, scheduler