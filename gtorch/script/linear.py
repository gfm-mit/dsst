import argparse
import sys

import gtorch.datasets.dataset
import gtorch.datasets.linear_box
import gtorch.datasets.linear_patient
import gtorch.hyper.coef
import gtorch.hyper.params
import gtorch.hyper.tune
import gtorch.models.linear_bc
import gtorch.models.linear_bnc
import gtorch.models.cnn_1d
import gtorch.models.cnn_1d_atrous
import gtorch.models.cnn_1d_butterfly
import gtorch.models.rnn_lstm
import gtorch.models.transformer
import gtorch.models.transformer_fft
import gtorch.optimize.metrics
import gtorch.optimize.optimizer
import util.excepthook
from plot.palette import draw_3_legends, get_3_axes, plot_3_types

if __name__ == "__main__":
  # Set the custom excepthook
  sys.excepthook = util.excepthook.custom_excepthook

  parser = argparse.ArgumentParser(description='Run a linear pytorch model')
  parser.add_argument('--coef', action='store_true', help='Plot coefficients')
  parser.add_argument('--tune', action='store_true', help='Tune parameters')
  parser.add_argument('--compare', action='store_true', help='Run on both linear_agg and linear')
  parser.add_argument('--device', default='cpu', help='torch device')
  parser.add_argument('--test', action='store_true', help='Train one batch on each model class')
  args = parser.parse_args()

  axs = None
  lines = []
  train_loader, val_loader, test_loader = gtorch.datasets.dataset.get_loaders()
  BUILDER = gtorch.models.transformer_fft.Transformer
  if args.test:
    for model_class, train_batch, val_batch in zip([
      gtorch.models.linear_bnc.Linear,
      gtorch.models.cnn_1d.Cnn,
      gtorch.models.cnn_1d_atrous.Cnn,
      gtorch.models.cnn_1d_butterfly.Cnn,
      gtorch.models.rnn_lstm.Rnn,
      gtorch.models.transformer.Transformer,
      gtorch.models.transformer_fft.Transformer,
    ], train_loader, val_loader):
      builder = model_class(n_classes=2, device=args.device)
      base_params = builder.get_parameters()
      base_params['min_epochs'] = 0
      base_params['max_epochs'] = 1
      retval, model = gtorch.hyper.params.setup_training_run(base_params, model_factory_fn=builder,
                                                             train_loader=[train_batch],
                                                             val_loader=[val_batch])
      print(model_class, retval)
  elif args.coef:
    # check coefficients
    gtorch.hyper.coef.get_coef_dist(
      builder=BUILDER(n_classes=2, device=args.device),
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader)
  elif args.tune:
    # tune parameters
    axs, line1 = gtorch.hyper.tune.main(train_loader, val_loader, test_loader, axs=axs, device=args.device, builder=BUILDER(n_classes=2, device=args.device))
  else:
    builder = BUILDER(n_classes=2, device=args.device)
    #torch.manual_seed(42)
    base_params = builder.get_parameters()
    #with cProfile.Profile() as pr:
    if True:  # just a placeholder for indentation
      retval, model = gtorch.hyper.params.setup_training_run(base_params, model_factory_fn=builder,
                                                            train_loader=train_loader, val_loader=val_loader)
    #pr.dump_stats('results/output_file.prof')
    model.eval()
    logits, targets = gtorch.optimize.metrics.get_combined_roc(model, test_loader, combine_fn=gtorch.datasets.linear_box.combiner)
    axs = get_3_axes() if axs is None else axs
    lines += [plot_3_types(logits, targets, axs)]
    if args.compare:
      train_loader, val_loader, test_loader = gtorch.datasets.dataset.get_loaders()
      builder = gtorch.models.linear_bnc.Linear(n_classes=2, device=args.device)
      #torch.manual_seed(42)
      base_params = builder.get_parameters()
      #base_params["max_epochs"] *= 100 # because less data
      retval, model = gtorch.hyper.params.setup_training_run(base_params, model_factory_fn=builder,
                                                             train_loader=train_loader, val_loader=val_loader)

      model.eval()
      logits, targets = gtorch.optimize.metrics.get_combined_roc(model, test_loader, combine_fn=gtorch.datasets.linear_box.combiner)
      axs = get_3_axes() if axs is None else axs
      lines += [plot_3_types(logits, targets, axs)]
    draw_3_legends(axs, lines)