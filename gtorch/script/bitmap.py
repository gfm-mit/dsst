import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import sys
import cProfile

from plot.palette import plot_3_types, get_3_axes, draw_3_legends
import gtorch.datasets.bitmap
import gtorch.models.cnn
import gtorch.optimize.optimize
import gtorch.hyper.params
import gtorch.hyper.tune
import util.excepthook

if __name__ == "__main__":
  # Set the custom excepthook
  sys.excepthook = util.excepthook.custom_excepthook

  parser = argparse.ArgumentParser(description='Run a linear pytorch model')
  parser.add_argument('--tune', action='store_true', help='Tune parameters')
  parser.add_argument('--device', default='cpu', help='torch device')
  args = parser.parse_args()

  axs = None
  lines = []
  train_loader, val_loader, test_loader = gtorch.datasets.bitmap.get_loaders()
  if args.tune:
    # tune parameters
    axs, line1 = gtorch.hyper.tune.main(train_loader, val_loader, test_loader, axs=axs, builder=gtorch.models.cnn.Cnn(n_classes=2, device=args.device, n_features=12))
  else:
    builder = gtorch.models.cnn.Cnn(n_classes=2, device=args.device, n_features=12)
    #torch.manual_seed(42)
    base_params = builder.get_parameters()
    with cProfile.Profile() as pr:
      retval, model = gtorch.hyper.params.setup_training_run(base_params, model_factory_fn=builder,
                                                             train_loader=train_loader, val_loader=val_loader)
    pr.dump_stats('results/output_file.prof')
    model.eval()
    logits, targets = gtorch.optimize.metrics.get_combined_roc(model, test_loader, combine_fn=gtorch.datasets.linear_box.combiner)
    axs = get_3_axes() if axs is None else axs
    lines += [plot_3_types(logits, targets, axs)]
    draw_3_legends(axs, lines)