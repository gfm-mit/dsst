import torch
import pytorch_optimizer.optimizer.sam
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, OPTIMIZER, PARAMETERS
from pytorch_optimizer.base.exception import NoClosureError

import models.bn_utils

class SAM(pytorch_optimizer.optimizer.sam.SAM):
  @torch.no_grad()
  def step(self, closure: CLOSURE = None):
    if closure is None:
      raise NoClosureError(str(self))

    with torch.enable_grad():
      loss = closure()
    self.first_step(zero_grad=True)

    # TODO: figure out how to do this using only params, not model
    models.bn_utils.disable_running_stats(self.params)
    with torch.enable_grad():
      closure()
    self.second_step()
    models.bn_utils.enable_running_stats(self.params)
    return loss