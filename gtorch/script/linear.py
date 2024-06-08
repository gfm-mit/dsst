import argparse
import sys
import cProfile
import util.excepthook

import gtorch.datasets.bitmap
import gtorch.datasets.dataset
import gtorch.datasets.linear_box
import gtorch.datasets.linear_patient
import gtorch.hyper.coef
import gtorch.hyper.experiment
import gtorch.hyper.params
import gtorch.hyper.tune
import gtorch.models.cnn_1d
import gtorch.models.cnn_1d_atrous
import gtorch.models.cnn_1d_butterfly
import gtorch.models.cnn_2d
import gtorch.models.linear_bc
import gtorch.models.linear_bnc
import gtorch.models.rnn_lstm
import gtorch.models.transformer
import gtorch.models.transformer_fft
import gtorch.optimize.loss
import gtorch.optimize.metrics
import gtorch.optimize.optimizer
from plot.palette import draw_3_legends

if __name__ == "__main__":
  # Set the custom excepthook
  sys.excepthook = util.excepthook.custom_excepthook

  parser = argparse.ArgumentParser(description='Run a linear pytorch model')
  parser.add_argument('--coef', action='store_true', help='Plot coefficients')
  parser.add_argument('--tune', action='store_true', help='Tune parameters')
  parser.add_argument('--profile', action='store_true', help='Profile training')
  parser.add_argument('--bitmap', action='store_true', help='Use bitmap data')
  parser.add_argument('--compare', action='store_true', help='Run on both linear_agg and linear')
  parser.add_argument('--device', default='cpu', help='torch device')
  parser.add_argument('--pretraining', default='none', help='whether to pretrain')
  parser.add_argument('--test', action='store_true', help='Train one batch on each model class')
  args = parser.parse_args()

  axs = None
  train_loader, val_loader, test_loader = gtorch.datasets.dataset.get_loaders()
  BUILDER = gtorch.models.linear_bnc.Linear
  if args.bitmap:
    BUILDER = gtorch.models.cnn_2d.Cnn
    train_loader, val_loader, test_loader = gtorch.datasets.bitmap.get_loaders()
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
      experiment = gtorch.hyper.experiment.Experiment(
        model_class=model_class,
        train_loader=[train_batch],
        val_loader=[val_batch],
        test_loader=None,
        args=args)
      retval = experiment.train(min_epochs=0, max_epochs=1)
      print(model_class, retval)
  elif args.coef:
    # check coefficients
    gtorch.hyper.coef.get_coef_dist(
      builder=BUILDER(n_classes=2, device=args.device),
      train_loader=train_loader,
      val_loader=val_loader)
  else:
    experiment = gtorch.hyper.experiment.Experiment(
      model_class=BUILDER,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      args=args)
    if args.tune:
      # tune parameters
      experiment.tune()
    if args.profile:
      with cProfile.Profile() as pr:
        experiment.train()
      pr.dump_stats('results/output_file.prof')
    else:
      axs, lines = None, None
      experiment.train()
      axs, lines = experiment.plot_trained(axs, lines)

      if args.compare:
        experiment = gtorch.hyper.experiment.Experiment(
          model_class=gtorch.models.linear_bnc.Linear,
          train_loader=train_loader,
          val_loader=val_loader,
          test_loader=test_loader,
          args=args)
        experiment.train()
        axs, lines = experiment.plot_trained(axs, lines)
      draw_3_legends(axs, lines)