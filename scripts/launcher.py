import argparse
import pathlib
import pstats
import sys
import cProfile
import os
import torch

import pandas as pd
import scipy
import sklearn
import sklearn.metrics
import models.cnn_2d
import models.linear_bnc
from plot.tune import print_and_plot_params
from util.config import TomlAction
import util.excepthook
import matplotlib.pyplot as plt
import numpy as np

import etl.torch.bitmap
import etl.torch.dataset
import etl.torch.linear_box
import etl.torch.linear_patient
import wrappers.coef
import wrappers.experiment
import plot.lr_finder
import plot.tune
import models.registry
import util.config

class LogAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
      assert isinstance(values, str)
      assert pathlib.Path(values).is_dir()
      setattr(namespace, self.dest, values)

def parse_args():
  parser = argparse.ArgumentParser(description='Run a linear pytorch model')
  parser.add_argument('--config', action=TomlAction, help='read a toml file specifying variants to run')

  parser.add_argument('--coef', action='store_true', help='plot linear coefficients')
  parser.add_argument('--profile', action='store_true', help='profile training with cProfile')
  parser.add_argument('--bitmap', action='store_true', help='use bitmap (2d) data and 2d CNN')

  parser.add_argument('--device', default='cpu', help='torch device')
  parser.add_argument('--model', default='', help='which model class to use')
  parser.add_argument('--task', default='classify', choices=set("next_token classify classify_patient classify_section".split()), help='training target / loss')
  parser.add_argument('--stats', default='train_loss', choices=set("train_loss thresholds epochs params".split()), help='output types to generate')

  parser.add_argument('--disk', default='none', choices=set("none load save freeze".split()), help='whether to persist the model (or use persisted)')
  parser.add_argument('--log', action='store_true', help='filename to log metrics and parameters')
  parser.add_argument('--offset', type=int, default=1, help='how far in advance to pretrain')
  args = parser.parse_args()
  return args

def main():
  sys.excepthook = util.excepthook.custom_excepthook
  # will save stack traces from creation in components, makes error messages less stupid
  #torch.autograd.set_detect_anomaly(True)
  #os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

  args = parse_args()

  if args.bitmap:
    args.model = "2d"
    BUILDER = models.cnn_2d.Cnn
    loader_fn = etl.torch.bitmap.get_loaders
    train_loader, val_loader, calibration_loader, test_loader = etl.torch.bitmap.get_loaders(device=args.device, task=args.task, batch_size=64)
  else:
    loader_fn = etl.torch.dataset.get_loaders
    train_loader, val_loader, calibration_loader, test_loader = etl.torch.dataset.get_loaders(device=args.device, task=args.task, batch_size=1000)
  if args.coef:
    # check coefficients
    BUILDER = models.linear_bnc.Linear
    wrappers.coef.get_coef_dist(
      builder=BUILDER(n_classes=2, device=args.device),
      train_loader=train_loader,
      val_loader=val_loader)
  else:
    BUILDER = models.linear_bnc.Linear
    experiment = wrappers.experiment.Experiment(
      model_class=BUILDER,
      loader_fn=loader_fn,
      train_loader=train_loader,
      val_loader=val_loader,
      calibration_loader=calibration_loader,
      test_loader=test_loader,
      args=args)
    if args.profile:
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
      if args.config:
        setups = util.config.get_setups(args.config)
      else:
        setups = {"<no config>": {}}
      plt.ion()
      y_axis_history = []
      tuning_history = []
      metric_history = []
      for i, (k, v) in enumerate(setups.items()):
        tqdm_prefix=f"Tuning[{i+1}/{len(setups)}]={k}"
        if 'batch' in v:
          experiment.redefine_loaders(v['batch'])
        metric, epoch_loss_history = experiment.train(tqdm_prefix=tqdm_prefix, **v)
        if args.stats == "params":
          logits, targets = experiment.batch_eval_test()
          tuning_history += [v]
          if args.task == "next_token":
            # dumb that sklearn RMSE doesn't work for 3 tensors
            rmse = np.sqrt(np.mean((targets - logits) ** 2))
            best_epoch = np.argmin(epoch_loss_history)
            metric_history += [dict(rmse=rmse, best_epoch=best_epoch)]
          else:
            plot_metric = sklearn.metrics.roc_auc_score(targets, logits)
            probs = scipy.special.expit(logits)
            brier = sklearn.metrics.brier_score_loss(targets, probs)
            best_epoch = np.argmin(epoch_loss_history)
            metric_history += [dict(auc=plot_metric, brier=brier, best_epoch=best_epoch)]
        elif args.stats in "train_loss epochs".split():
          pd.Series(epoch_loss_history).to_csv(f"results/epoch/{i}.csv")
          axs = plot.tune.plot_history(args, epoch_loss_history, axs=axs, label=k)
          y_axis_history += [epoch_loss_history]
        #elif args.task in 'classify classify_patient classify_section'.split():
        elif args.stats == "thresholds":
          axs = experiment.plot_trained(axs, label=k)
        else:
          assert False
        plt.pause(0.1)
      plt.ioff()
      if args.stats == "params":
        tuning_history = pd.DataFrame(tuning_history)
        metric_history = pd.DataFrame(metric_history)
        print_and_plot_params(args, tuning_history, metric_history)
        return
      if args.stats in "train_loss epochs".split():
        plot.tune.set_ylim(np.concatenate(y_axis_history))
      suptitle = "Aggregated at the Box Level, not Patient" if args.task == "classify" else "Aggregated at Patient Level, not Box"
      plt.suptitle(suptitle)
      try:
        plt.tight_layout()
      except OverflowError:
        print("OverflowError in tight_layout")
      plt.show()

if __name__ == "__main__":
  main()