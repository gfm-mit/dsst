import torch
import pytorch_optimizer

import gtorch.loss.scheduler


def get_optimizer_and_scheduler(params, model):
  optimizer = get_optimizer(params, model)
  return gtorch.loss.scheduler.get_scheduler(params, model, optimizer)

def get_optimizer(params, model):
  if "optimizer" in params and params["optimizer"] == "adam":
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params["learning_rate"],
                                  betas=[
                                      params["momentum"],
                                      params["conditioning_smoother"],
                                  ],
                                  weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "signedmomentum":
    optimizer = pytorch_optimizer.Lion(model.parameters(),
                                       betas=[
                                           params["momentum"],
                                           params["conditioning_smoother"],
                                       ],
                                       weight_decouple=True,
                                       weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "prodigy":
    optimizer = pytorch_optimizer.Prodigy(model.parameters(),
                                          betas=[
                                              params["momentum"],
                                              params["conditioning_smoother"],
                                          ],
                                          weight_decouple=True,
                                          weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "sfsgd":
    optimizer = pytorch_optimizer.ScheduleFreeSGD(model.parameters(),
                                                  lr=params["learning_rate"],
                                                  momentum=params["momentum"],
                                                  weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "samsgd":
    optimizer = pytorch_optimizer.SAM(model.parameters(),
                                      base_optimizer=torch.optim.SGD,
                                      lr=params["learning_rate"],
                                      momentum=params["momentum"],
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
  return optimizer