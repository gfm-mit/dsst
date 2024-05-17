import pandas as pd
import torch
from tqdm import tqdm

from etl.parse_semantics import *
from etl.parse_dynamics import *

from plot.palette import *
import gtorch.datasets.linear_agg
import gtorch.datasets.linear
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params
import gtorch.hyper.tune
import util.excepthook
import sys

def main(train_loader, val_loader, test_loader, axs=None, device='cpu'):
  #torch.manual_seed(42)
  model, base_params = gtorch.models.linear.get_model(hidden_width=2, device=device, classes=2)
  retval, model = gtorch.hyper.params.many_hyperparams(base_params, model_factory_fn=gtorch.models.linear.get_model,
                                                       train_loader=train_loader, val_loader=val_loader)
  results = []
  model.eval()
  with torch.no_grad():
    for idx, (data, target, g) in enumerate(test_loader):
      #print(target)
      output = model(data.to(device)).to('cpu')
      results += [pd.DataFrame(dict(
          logits=output.detach().numpy()[:, 1],
          targets=target.detach().to('cpu').numpy()[:, 0],
          groups=g,
      ))]
      if idx % 100 == 0:
        print(idx)
  # TODO: why is this thing not working at all?
  df = pd.concat(results)
  df = df.groupby("groups").mean()
  logits, targets = df.logits.values, df.targets.values

  axs = get_3_axes() if axs is None else axs
  line1 = plot_3_types(logits, targets, axs)
  return axs, line1
  draw_3_legends(axs, [line1])

if __name__ == "__main__":
  # Set the custom excepthook
  sys.excepthook = util.excepthook.custom_excepthook
  axs = None
  train_loader, val_loader, test_loader = gtorch.datasets.linear_agg.get_loaders()
  # TODO: evil!  val and test on train set
  axs, line1 = main(train_loader, val_loader, test_loader, axs=axs, device='cpu')
  #train_loader, val_loader, test_loader = gtorch.datasets.linear.get_loaders()
  #axs, line2 = main(train_loader, val_loader, test_loader, axs=axs)
  #draw_3_legends(axs, [line1, line2])
  draw_3_legends(axs, [line1])
  #cProfile.run('main()', 'output_file.prof')