import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import sys
import cProfile

from plot.palette import plot_3_types, get_3_axes, draw_3_legends
import gtorch.datasets.synthetic
import gtorch.datasets.linear_agg
import gtorch.datasets.linear
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params
import gtorch.hyper.tune
import gtorch.hyper.coef
import util.excepthook

def get_roc(model, test_loader, axs=None, device='cpu'):
  results = []
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

  parser = argparse.ArgumentParser(description='Run a linear pytorch model')
  parser.add_argument('--coef', action='store_true', help='Plot coefficients')
  parser.add_argument('--tune', action='store_true', help='Tune parameters')
  args = parser.parse_args()

  axs = None
  lines = []
  train_loader, val_loader, test_loader = gtorch.datasets.linear_agg.get_loaders()
  if args.coef:
    # check coefficients
    gtorch.hyper.coef.get_coef_dist(
      builder=gtorch.models.linear.Linear(classes=2),
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader)
  elif args.tune:
    # tune parameters
    axs, line1 = gtorch.hyper.tune.main(train_loader, val_loader, test_loader, axs=axs, device='cpu', builder=gtorch.models.linear.Linear(classes=2))
  else:
    # just train a model and display ROC plots
    builder = gtorch.models.linear.Linear(classes=2)
    #torch.manual_seed(42)
    base_params = builder.get_parameters()
    #with cProfile.Profile() as pr:
    retval, model = gtorch.hyper.params.setup_training_run(base_params, model_factory_fn=builder,
                                                        train_loader=train_loader, val_loader=val_loader)

    #pr.dump_stats('results/output_file.prof')
    model.eval()
    axs, line2 = get_roc(model, test_loader, axs=axs)
    lines += [line2]
    draw_3_legends(axs, lines)