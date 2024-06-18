import argparse
import pathlib
import sys
import cProfile
import util.excepthook
import tomli
import matplotlib.pyplot as plt

import gtorch.datasets.bitmap
import gtorch.datasets.dataset
import gtorch.datasets.linear_box
import gtorch.datasets.linear_patient
import gtorch.train.coef
import gtorch.train.experiment
import gtorch.train.train
import gtorch.train.tune
import plot.lr_finder
import plot.tune
import gtorch.models.cnn_1d
import gtorch.models.cnn_1d_atrous
import gtorch.models.cnn_1d_butterfly
import gtorch.models.cnn_2d
import gtorch.models.linear_bc
import gtorch.models.linear_bnc
import gtorch.models.rnn_lstm
import gtorch.models.transformer
import gtorch.models.registry
import gtorch.loss.loss
import gtorch.metrics.metrics
import gtorch.loss.optimizer

class TomlAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
      assert isinstance(values, str)
      key = None
      if "," in values:
        values, key = values.split(",")
      with open(values, 'rb') as stream:
        config = tomli.load(stream)
      if key is not None:
        config = config[key]
      setattr(namespace, self.dest, config)

class LogAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
      if values == "":
        return
      assert isinstance(values, str)
      #assert not pathlib.Path(values).exists()
      pathlib.Path(values).touch()
      setattr(namespace, self.dest, values)

if __name__ == "__main__":
  sys.excepthook = util.excepthook.custom_excepthook
  # will save stack traces from creation in components, makes error messages less stupid
  #torch.autograd.set_detect_anomaly(True)

  parser = argparse.ArgumentParser(description='Run a linear pytorch model')
  parser.add_argument('--coef', action='store_true', help='Plot coefficients')
  parser.add_argument('--tune', action='store_true', help='Tune parameters')
  parser.add_argument('--profile', action='store_true', help='Profile training')
  parser.add_argument('--bitmap', action='store_true', help='Use bitmap data')
  parser.add_argument('--compare', action='store_true', help='Run on both linear_agg and linear')
  parser.add_argument('--find_lr', action='store_true', help='Use CLR method to find learning rate')
  parser.add_argument('--history', default='none', choices=set("none train val".split()), help='Plot history of loss')
  parser.add_argument('--device', default='cpu', help='torch device')
  parser.add_argument('--test', action='store_true', help='Train one batch on each model class')
  parser.add_argument('--model', default='linear', help='which model class to use')
  parser.add_argument('--task', default='classify', choices=set("next_token classify classify_patient".split()), help='training target / loss')
  parser.add_argument('--disk', default='none', choices=set("none load save".split()), help='whether to persist the model (or use persisted)')
  parser.add_argument('--log', action=LogAction, default='', help='filename to log metrics and parameters')
  parser.add_argument('--config', action=TomlAction, default={}, help='read a config toml file')
  args = parser.parse_args()

  axs = None
  train_loader, val_loader, test_loader = gtorch.datasets.dataset.get_loaders()
  BUILDER = gtorch.models.registry.lookup_model(args.model)
  if args.bitmap:
    BUILDER = gtorch.models.cnn_2d.Cnn
    train_loader, val_loader, test_loader = gtorch.datasets.bitmap.get_loaders()
  if args.test:
    for model_class, train_batch, val_batch in zip(
      gtorch.models.registry.get_all_1d_models(),
      train_loader, val_loader):
      experiment = gtorch.train.experiment.Experiment(
        model_class=model_class,
        train_loader=[train_batch],
        val_loader=[val_batch],
        test_loader=None,
        args=args)
      print(f"starting test {model_class=}")
      metric, epoch_loss_history = experiment.train(min_epochs=0, max_epochs=1)
      print(f"finished test {model_class=} {metric=}")
  elif args.coef:
    # check coefficients
    gtorch.train.coef.get_coef_dist(
      builder=BUILDER(n_classes=2, device=args.device),
      train_loader=train_loader,
      val_loader=val_loader)
  else:
    experiment = gtorch.train.experiment.Experiment(
      model_class=BUILDER,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      args=args)
    if args.tune:
      # tune parameters
      experiment.tune()
    elif args.find_lr:
      if args.disk == "save":
        metric, epoch_loss_history = experiment.train(scheduler=None, **args.config)
        axs = plot.tune.plot_epoch_loss_history(args, epoch_loss_history)
        plt.show()
      else:
        experiment.find_momentum(momentum=[0, 0.5, 0.9])
    elif args.profile:
      with cProfile.Profile() as pr:
        experiment.train()
      pr.dump_stats('results/output_file.prof')
    else:
      axs = None
      if args.compare:
        setups = args.config
      else:
        setups = {"base": args.config["base"] if "base" in args.config else {}}
      plt.ion()
      for k, v in setups.items():
        metric, epoch_loss_history = experiment.train(**v)
        if args.history != "none":
          axs = plot.tune.plot_epoch_loss_history(args, epoch_loss_history, axs=axs, label=k)
        elif args.task in 'classify classify_patient'.split():
          axs = experiment.plot_trained(axs, label=k)
        plt.pause(0.1)
      plt.ioff()
      suptitle = "Aggregated at the Box Level, not Patient" if args.task == "classify" else "Aggregated at Patient Level, not Box"
      plt.suptitle(suptitle)
      plt.tight_layout()
      plt.show()