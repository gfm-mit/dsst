import argparse
import cProfile
import sys

import gtorch.datasets.bitmap
import gtorch.hyper.params
import gtorch.hyper.tune
import gtorch.models.cnn_2d
import gtorch.optimize.optimizer
import util.excepthook
from plot.palette import draw_3_legends, get_3_axes, plot_3_types

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
    BUILDER = gtorch.models.cnn_2d.Cnn(n_classes=2, device=args.device, n_features=12)
    base_params = BUILDER.get_parameters()
    axs, line1 = gtorch.hyper.tune.main(train_loader, val_loader,
                                        base_params=base_params,
                                        builder=BUILDER)
  else:
    builder = gtorch.models.cnn_2d.Cnn(n_classes=2, device=args.device, n_features=12)
    #torch.manual_seed(42)
    base_params = builder.get_parameters()
    with cProfile.Profile() as pr:
      retval, train_loss, model = gtorch.hyper.params.setup_training_run(base_params, model_factory_fn=builder,
                                                             train_loader=train_loader, val_loader=val_loader)
    pr.dump_stats('results/output_file.prof')
    model.eval()
    logits, targets = gtorch.optimize.metrics.get_combined_roc(model, test_loader, combine_fn=gtorch.datasets.linear_box.combiner)
    axs = get_3_axes() if axs is None else axs
    lines += [plot_3_types(logits, targets, axs)]
    draw_3_legends(axs, lines)