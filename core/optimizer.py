import torch
import pytorch_optimizer

import core.scheduler


def get_optimizer_and_scheduler(params, model):
  optimizer = get_optimizer(params, model)
  return core.scheduler.get_scheduler(params, model, optimizer)

def get_optimizer(params, model):
  assert "optimizer" in params
  if params["optimizer"] == "adam":
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params["learning_rate"],
                                  betas=[
                                      params["momentum"],
                                      params["conditioning_smoother"],
                                  ],
                                  weight_decay=params["weight_decay"])
  elif params["optimizer"] == "lion":
    # this thing is just the signed momentum optimizer, rediscovered by GDM and then published because life is fair
    # needs huge model / batch size to counteract the variance
    # because it preconditions by just taking the sign of the update, which is crazy
    # but saves 30% of memory
    optimizer = pytorch_optimizer.Lion(model.parameters(),
                                       betas=[
                                           params["momentum"],
                                           params["conditioning_smoother"],
                                       ],
                                       weight_decouple=True,
                                       weight_decay=params["weight_decay"])
  elif params["optimizer"] == "prodigy":
    optimizer = pytorch_optimizer.Prodigy(model.parameters(),
                                          betas=[
                                              params["momentum"],
                                              params["conditioning_smoother"],
                                          ],
                                          growth_rate=1.3,
                                          weight_decouple=True,
                                          weight_decay=params["weight_decay"])
  elif params["optimizer"] == "sfsgd":
    optimizer = pytorch_optimizer.ScheduleFreeSGD(model.parameters(),
                                                  lr=params["learning_rate"],
                                                  momentum=params["momentum"],
                                                  weight_decay=params["weight_decay"])
  elif params["optimizer"] == "sfadam":
    optimizer = pytorch_optimizer.ScheduleFreeAdamW(model.parameters(),
                                                    lr=params["learning_rate"],
                                                    betas=[
                                                        params["momentum"],
                                                        params["conditioning_smoother"],
                                                    ],
                                                    weight_decay=params["weight_decay"])
  elif params["optimizer"] == "asamsgd":
    optimizer = pytorch_optimizer.SAM(model.parameters(),
                                      base_optimizer=torch.optim.SGD,
                                      adaptive=True,
                                      lr=params["learning_rate"],
                                      momentum=params["momentum"],
                                      weight_decay=params["weight_decay"])
  elif params["optimizer"] == "samsgd":
    optimizer = pytorch_optimizer.SAM(model.parameters(),
                                      base_optimizer=torch.optim.SGD,
                                      lr=params["learning_rate"],
                                      momentum=params["momentum"],
                                      weight_decay=params["weight_decay"])
  elif params["optimizer"] == "asamadam":
    optimizer = pytorch_optimizer.SAM(model.parameters(),
                                      base_optimizer=torch.optim.AdamW,
                                      adaptive=True,
                                      lr=params["learning_rate"],
                                      betas=[
                                          params["momentum"],
                                          params["conditioning_smoother"],
                                      ],
                                      weight_decay=params["weight_decay"])
  elif params["optimizer"] == "samadam":
    optimizer = pytorch_optimizer.SAM(model.parameters(),
                                      base_optimizer=torch.optim.AdamW,
                                      lr=params["learning_rate"],
                                      betas=[
                                          params["momentum"],
                                          params["conditioning_smoother"],
                                      ],
                                      weight_decay=params["weight_decay"])
  elif params["optimizer"] == "sgd":
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params["learning_rate"],
                                momentum=params["momentum"],
                                weight_decay=params["weight_decay"])
  else:
    assert False, f"unknown optimizer {params['optimizer']}, defaulting to sgd"
  return optimizer