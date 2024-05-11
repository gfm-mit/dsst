import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import torch

from etl.parse_semantics import *
from etl.parse_dynamics import *

from plot.palette import *
import gtorch.datasets.bitmap
import gtorch.models.cnn
import gtorch.optimize.optimize
import gtorch.hyper.params
import cProfile

def main(train_loader, val_loader, axs=None):
  model, base_params = gtorch.models.cnn.get_model(hidden_width=4, device='cpu', classes=1)
  base_params["max_epochs"] = 3
  #r = q(next(iter(train_loader))[0].to('cpu'))
  #gtorch.optimize.optimize.optimize(0, q, gtorch.optimize.optimize.FakeOptimizer(q), train_loader)
  #loss, accuracy = gtorch.optimize.optimize.metrics(q, val_loader=train_loader)
  retval, model = gtorch.hyper.params.many_hyperparams(base_params, model_factory_fn=gtorch.models.cnn.get_model,
                                                       train_loader=train_loader, val_loader=val_loader)

  DEVICE = 'cpu'
  results = []
  model.eval()
  with torch.no_grad():
    for idx, (data, target) in enumerate(val_loader):
      #print(target)
      output = model(data.to(DEVICE)).to('cpu')
      results += [(
          output.detach().numpy(),
          target.detach().to('cpu').numpy(),
      )]
      if idx % 100 == 0:
        print(idx)
  logits, targets = zip(*results)

  axs = get_3_axes() if axs is None else axs
  logits = logits[0][:, 0]
  targets = targets[0][:, 0]
  #print(pd.DataFrame(dict(logits=logits, targets=targets)).head())
  line1 = plot_3_types(logits, targets, axs)
  return axs, line1
  draw_3_legends(axs, [line1])

if __name__ == "__main__":
  axs = None
  train_loader = gtorch.datasets.bitmap.get_loaders()
  val_loader = train_loader #TODO: evil!!
  axs, line1 = main(train_loader, val_loader, axs=axs)
  train_loader = gtorch.datasets.bitmap.get_loaders()
  val_loader = train_loader #TODO: evil!!
  axs, line2 = main(train_loader, val_loader, axs=axs)
  draw_3_legends(axs, [line1, line2])
  #cProfile.run('main()', 'output_file.prof')