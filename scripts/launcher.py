import argparse
import pathlib
import pstats
import sys
import cProfile

import pandas as pd
import scipy
import sklearn
import models.cnn_2d
import models.linear_bnc
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
import wrappers.tune
import plot.lr_finder
import plot.tune
import plot.history
import models.registry
import util.config

class LogAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
      assert isinstance(values, str)
      assert pathlib.Path(values).is_dir()
      setattr(namespace, self.dest, values)

def parse_args():
  parser = argparse.ArgumentParser(description='Run a linear pytorch model')
  parser.add_argument('--config', action=TomlAction, help='read a config toml file')

  parser.add_argument('--coef', action='store_true', help='Plot coefficients')
  parser.add_argument('--profile', action='store_true', help='Profile training')
  parser.add_argument('--bitmap', action='store_true', help='Use bitmap data')

  parser.add_argument('--device', default='cpu', help='torch device')
  parser.add_argument('--model', default='linear', help='which model class to use')
  parser.add_argument('--task', default='classify', choices=set("next_token classify classify_patient classify_section".split()), help='training target / loss')
  parser.add_argument('--history', default='none', choices=set("none train val variance".split()), help='Plot history of loss')

  parser.add_argument('--disk', default='none', choices=set("none load save freeze".split()), help='whether to persist the model (or use persisted)')
  parser.add_argument('--log', action=LogAction, default='', help='filename to log metrics and parameters')
  parser.add_argument('--offset', type=int, default=1, help='how far in advance to pretrain')
  args = parser.parse_args()
  return args

def main():
  sys.excepthook = util.excepthook.custom_excepthook
  # will save stack traces from creation in components, makes error messages less stupid
  import torch
  torch.autograd.set_detect_anomaly(True)

  args = parse_args()

  if args.bitmap:
    args.model = "2d"
    BUILDER = models.cnn_2d.Cnn
    train_loader, val_loader, calibration_loader, test_loader = etl.torch.bitmap.get_loaders(device=args.device, task=args.task)
  else:
    train_loader, val_loader, calibration_loader, test_loader = etl.torch.dataset.get_loaders(device=args.device, task=args.task)
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
        metric, epoch_loss_history = experiment.train(tqdm_prefix=tqdm_prefix, **v)
        if args.log != "":
          experiment.log_training(epoch_loss_history, k)
        if args.history == "variance":
          logits, targets = experiment.batch_eval_test()
          auc = sklearn.metrics.roc_auc_score(targets, logits)
          probs = scipy.special.expit(logits)
          brier = sklearn.metrics.brier_score_loss(targets, probs)
          metric_history += [dict(auc=auc, brier=brier)]
          tuning_history += [v]
        elif args.history in "train val".split():
          axs = plot.history.plot_history(args, epoch_loss_history, axs=axs, label=k)
          y_axis_history += [epoch_loss_history]
        elif args.task in 'classify classify_patient classify_section'.split():
          axs = experiment.plot_trained(axs, label=k)
        plt.pause(0.1)
      plt.ioff()
      if args.history in "train val".split():
        plot.tune.set_ylim(np.concatenate(y_axis_history))
      elif args.history == "variance":
        #print(metric_history)
        metrics = pd.DataFrame(metric_history)
        #print(metrics)
        auc = metrics.auc.copy()
        #print(auc)
        metrics = metrics.aggregate(["mean", "std"], axis=0).transpose()
        print(" & ".join(metrics.columns))
        metrics = [
          f"${v['mean']:.3f} \pm {v['std']:.3f}$"
          for _, v in metrics.iterrows()
        ]
        print(" & ".join(metrics))
        plot.history.plot_best_values(pd.DataFrame(tuning_history), auc, task=args.task)
        exit(0)
      suptitle = "Aggregated at the Box Level, not Patient" if args.task == "classify" else "Aggregated at Patient Level, not Box"
      plt.suptitle(suptitle)
      try:
        plt.tight_layout()
      except OverflowError:
        print("OverflowError in tight_layout")
      plt.show()

if __name__ == "__main__":
  main()