import argparse
import pathlib
import pstats
import sys
import cProfile
import util.excepthook
import tomli
import matplotlib.pyplot as plt
import numpy as np

import etl.torch.bitmap
import etl.torch.dataset
import etl.torch.linear_box
import etl.torch.linear_patient
import wrappers.coef
import wrappers.experiment
import wrappers.tune
import plot.lr_finder
import plot.tune
import models.registry

class TomlAction(argparse.Action):
    def __init__(self, toml_key=None, **kwargs):
        self.key = toml_key
        assert "default" not in kwargs
        kwargs["default"] = {}
        super(TomlAction, self).__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
      assert isinstance(values, str)
      with open(values, 'rb') as stream:
        config = tomli.load(stream)
      if self.key is not None:
        config = config[self.key]
      setattr(namespace, self.dest, config)

class LogAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
      assert isinstance(values, str)
      assert pathlib.Path(values).is_dir()
      setattr(namespace, self.dest, values)

def parse_args():
  parser = argparse.ArgumentParser(description='Run a linear pytorch model')
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--tune', action=TomlAction, toml_key='tune', help='Parameter ranges to tune')
  group.add_argument('--compare', action=TomlAction, toml_key='compare', help='Run multiple prespecified configs')
  group.add_argument('--find_lr', action=TomlAction, toml_key='find_lr', help='Config file for CLR method to find learning rate')
  group.add_argument('--config', action=TomlAction, help='read a config toml file')
  group.add_argument('--test', action='store_true', help='Train one batch on each model class')

  parser.add_argument('--coef', action='store_true', help='Plot coefficients')
  parser.add_argument('--profile', action='store_true', help='Profile training')
  parser.add_argument('--bitmap', action='store_true', help='Use bitmap data')

  parser.add_argument('--device', default='cpu', help='torch device')
  parser.add_argument('--model', default='linear', help='which model class to use')
  parser.add_argument('--task', default='classify', choices=set("next_token classify classify_patient".split()), help='training target / loss')
  parser.add_argument('--history', default='none', choices=set("none train val".split()), help='Plot history of loss')

  parser.add_argument('--disk', default='none', choices=set("none load save".split()), help='whether to persist the model (or use persisted)')
  parser.add_argument('--log', action=LogAction, default='', help='filename to log metrics and parameters')
  args = parser.parse_args()
  return args

def main():
  sys.excepthook = util.excepthook.custom_excepthook
  # will save stack traces from creation in components, makes error messages less stupid
  #torch.autograd.set_detect_anomaly(True)

  args = parse_args()

  if args.bitmap:
    BUILDER = models.cnn_2d.Cnn
    train_loader, val_loader, test_loader = etl.torch.bitmap.get_loaders()
  else:
    train_loader, val_loader, test_loader = etl.torch.dataset.get_loaders(device=args.device)
    BUILDER = models.registry.lookup_model(args.model)
  if args.test:
    for model_class, train_batch, val_batch in zip(
      models.registry.get_all_1d_models(),
      train_loader, val_loader):
      experiment = wrappers.experiment.Experiment(
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
    wrappers.coef.get_coef_dist(
      builder=BUILDER(n_classes=2, device=args.device),
      train_loader=train_loader,
      val_loader=val_loader)
  else:
    use_experiment(args, BUILDER, train_loader, val_loader, test_loader)

def use_experiment(args, BUILDER, train_loader, val_loader, test_loader):
    experiment = wrappers.experiment.Experiment(
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
        plot.tune.plot_epoch_loss_history(args, epoch_loss_history)
        plt.show()
      else:
        experiment.find_momentum(momentum=[0.9])
    elif args.profile:
      with cProfile.Profile() as pr:
        experiment.train()
      pr.dump_stats('results/output_file.prof')
      stats = pstats.Stats('results/output_file.prof')
      stats.sort_stats('cumulative')
      stats.print_stats(30)
    else:
      compare(args, experiment)

def compare(args, experiment):
      axs = None
      if args.compare:
        setups = args.compare
      else:
        setups = {"base": args.config["base"] if "base" in args.config else {}}
      plt.ion()
      running_loss_history = []
      for k, v in setups.items():
        metric, epoch_loss_history = experiment.train(**v)
        if args.log != "":
          experiment.log_training(epoch_loss_history, k)
        if args.history != "none":
          axs = plot.tune.plot_epoch_loss_history(args, epoch_loss_history, axs=axs, label=k)
          running_loss_history += [epoch_loss_history]
        elif args.task in 'classify classify_patient'.split():
          axs = experiment.plot_trained(axs, label=k)
        plt.pause(0.1)
      plt.ioff()
      if args.history != "none":
        plot.tune.set_ylim(np.concatenate(running_loss_history))
      suptitle = "Aggregated at the Box Level, not Patient" if args.task == "classify" else "Aggregated at Patient Level, not Box"
      plt.suptitle(suptitle)
      plt.tight_layout()
      plt.show()

if __name__ == "__main__":
  main()