import torch
from pytorch_optimizer import DAdaptLion, Prodigy

def optimize(epoch, model, optimizer, train_loader):
  pass

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
  elif "schedule" in params and params["schedule"] == "exponential":
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=params["schedule_gamma"])
  else:
    scheduler = FakeOptimizer(model)
  return optimizer, scheduler