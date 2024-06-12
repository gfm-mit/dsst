import torch
import pytorch_optimizer

import gtorch.optimize.scheduler


def get_optimizer(params, model):
  optimizer = get_optimizer_(params, model)
  return gtorch.optimize.scheduler.get_scheduler(params, model, optimizer)

def get_optimizer_(params, model):
  if "optimizer" in params and params["optimizer"] == "adam":
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params["learning_rate"],
                                  betas=[
                                      params["momentum"],
                                      params["beta2"],
                                  ],
                                  weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "lion":
    optimizer = pytorch_optimizer.Lion(model.parameters(),
                                       betas=[
                                           params["momentum"],
                                           params["beta2"],
                                       ],
                                       weight_decouple=True,
                                       weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "dadaptlion":
    optimizer = pytorch_optimizer.DAdaptLion(model.parameters(),
                                             betas=[
                                                 params["momentum"],
                                                 params["beta2"],
                                             ],
                                             weight_decouple=True,
                                             weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "prodigy":
    optimizer = pytorch_optimizer.Prodigy(model.parameters(),
                                          betas=[
                                              params["momentum"],
                                              params["beta2"],
                                          ],
                                          weight_decouple=True,
                                          weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "sfsgd":
    optimizer = pytorch_optimizer.ScheduleFreeSGD(model.parameters(),
                                                  lr=params["learning_rate"],
                                                  momentum=params["momentum"],
                                                  weight_decay=params["weight_decay"])
  elif "optimizer" in params and params["optimizer"] == "sam":
    optimizer = pytorch_optimizer.SAM(model.parameters(),
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