import argparse
import cProfile
import sys
from matplotlib import pyplot as plt

import gtorch.datasets.bitmap
import gtorch.train.train
import gtorch.train.tune
import gtorch.metrics.calibration
import gtorch.metrics.metrics
import gtorch.models.cnn_2d
import gtorch.loss.optimizer
import plot.metrics
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
    BUILDER = gtorch.models.cnn_2d.Cnn(n_classes=2, device=args.device, n_features=12)
    base_params = BUILDER.get_parameters()
    axs, line1 = gtorch.train.tune.main(train_loader, val_loader,
                                        base_params=base_params,
                                        builder=BUILDER)
  else:
    builder = gtorch.models.cnn_2d.Cnn(n_classes=2, device=args.device, n_features=12)
    #torch.manual_seed(42)
    base_params = builder.get_parameters()
    with cProfile.Profile() as pr:
      retval, train_loss, model = gtorch.train.train.setup_training_run(base_params, model_factory_fn=builder,
                                                             train_loader=train_loader, val_loader=val_loader)
    pr.dump_stats('results/output_file.prof')
    model.eval()
    logits, targets = gtorch.metrics.metrics.get_combined_roc(model, test_loader, combine_fn=gtorch.datasets.linear_box.combiner)

    roc = gtorch.metrics.calibration.get_full_roc_table(logits, targets)
    axs = plot.metrics.plot_palette(roc, axs)
    plt.suptitle("linear_box.combiner")
    plt.tight_layout()
    plt.show()