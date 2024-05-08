import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import torch

from etl.parse_semantics import *
from etl.parse_dynamics import *

from plot.palette import *
import gtorch.datasets.linear
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params

if __name__ == "__main__":
  train_loader = gtorch.datasets.linear.get_loaders()
  val_loader = train_loader #TODO: evil!!
  model, base_params = gtorch.models.linear.get_model(hidden_width=2, device='cpu', classes=1)
  #r = q(next(iter(train_loader))[0].to('cpu'))
  #gtorch.optimize.optimize.optimize(0, q, gtorch.optimize.optimize.FakeOptimizer(q), train_loader)
  #loss, accuracy = gtorch.optimize.optimize.metrics(q, val_loader=train_loader)
  retval, model = gtorch.hyper.params.many_hyperparams(base_params, model_factory_fn=gtorch.models.linear.get_model,
                                                       train_loader=train_loader, val_loader=val_loader)

  DEVICE = 'cpu'
  results = []
  model.eval()
  with torch.no_grad():
    for data, target in val_loader:
      print(target)
      output = model(data.to(DEVICE)).to('cpu')
      results += [(
          output.detach().numpy(),
          target.detach().to('cpu').numpy(),
      )]
  logits, targets = zip(*results)

  axs = get_3_axes()
  logits = logits[0][:, 0]
  targets = targets[0][:, 0]
  print(pd.DataFrame(dict(logits=logits, targets=targets)).head())
  line1 = plot_3_types(logits, targets, axs)
  draw_3_legends(axs, [line1])