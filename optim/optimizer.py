import torch
import pytorch_optimizer

import optim.scheduler
import optim.rk4


def get_optimizer_and_scheduler(params, model):
  optimizer = get_optimizer(params, model)
  return optim.scheduler.get_scheduler(params, model, optimizer)

def get_optimizer(params, model):
  assert "optimizer" in params
  if "effective_lr_batch" in params:
    params["learning_rate"] *= params["effective_lr_batch"] / params["batch"]
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
    assert params["scheduler"] == "none"
    optimizer = pytorch_optimizer.Prodigy(model.parameters(),
                                          betas=[
                                              params["momentum"],
                                              params["conditioning_smoother"],
                                          ],
                                          lr=1e-1, # hack for --model=causal
                                          weight_decouple=True,
                                          weight_decay=params["weight_decay"])
  elif params["optimizer"] == "samprodigy":
    assert params["scheduler"] == "none"
    optimizer = pytorch_optimizer.SAM(model.parameters(),
                                      base_optimizer=pytorch_optimizer.Prodigy,
                                      lr=1,
                                      betas=[
                                          params["momentum"],
                                          params["conditioning_smoother"],
                                      ],
                                      weight_decay=params["weight_decay"])
  elif params["optimizer"] == "sfsgd":
    assert params["scheduler"] == "none"
    optimizer = pytorch_optimizer.ScheduleFreeSGD(model.parameters(),
                                                  lr=params["learning_rate"],
                                                  momentum=params["momentum"],
                                                  warmup_steps=params["warmup_epochs"],
                                                  weight_decay=params["weight_decay"])
  elif params["optimizer"] == "sfadam":
    assert params["scheduler"] == "none"
    optimizer = pytorch_optimizer.ScheduleFreeAdamW(model.parameters(),
                                                    lr=params["learning_rate"],
                                                    betas=[
                                                        params["momentum"],
                                                        params["conditioning_smoother"],
                                                    ],
                                                    warmup_steps=params["warmup_epochs"],
                                                    weight_decay=params["weight_decay"])
  elif params["optimizer"] == "samsgd":
    optimizer = pytorch_optimizer.SAM(model.parameters(),
                                      base_optimizer=torch.optim.SGD,
                                      lr=params["learning_rate"],
                                      momentum=params["momentum"],
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
  elif params["optimizer"] == "rk4":
    optimizer = optim.rk4.RK4(model.parameters(), named=model.named_parameters(), lr=params["learning_rate"])
  elif params["optimizer"] == "sgd":
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params["learning_rate"],
                                momentum=params["momentum"],
                                weight_decay=params["weight_decay"])
  else:
    assert False, f"unknown optimizer {params['optimizer']}, defaulting to sgd"
  return optimizer