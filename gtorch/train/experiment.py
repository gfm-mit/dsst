import gtorch.datasets.linear_box
import gtorch.train.lr_finder
import gtorch.train.lr_plots
import gtorch.train.train
import gtorch.train.tune
import gtorch.metrics.metrics
from plot.palette import get_3_axes, plot_3_types

class Experiment:
  def __init__(self, model_class, train_loader, val_loader, test_loader, args):
    self.model_class = model_class
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.args = args
    self.model = None

  def train(self, **kwargs):
    #torch.manual_seed(42)
    builder = self.model_class(n_classes=2, device=self.args.device)
    base_params = builder.get_parameters(task=self.args.task) | self.args.config | kwargs
    metric, epoch_loss_history, self.model = gtorch.train.train.setup_training_run(
        base_params, model_factory_fn=builder,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        task=self.args.task,
        disk=self.args.disk,
        history=self.args.history)
    return metric, epoch_loss_history

  def plot_trained(self, axs, lines):
    assert self.model is not None
    assert self.args.task in 'classify classify_patient'.split()
    self.model.eval()
    logits, targets = gtorch.metrics.metrics.get_combined_roc(
      self.model, self.test_loader,
      combine_fn=None if self.args.task == "classify" else gtorch.datasets.linear_box.combiner)
    axs = get_3_axes() if axs is None else axs
    lines = [] if lines is None else lines
    #import pandas as pd
    #pd.DataFrame(dict(logits=logits, targets=targets)).to_csv("results/roc.csv")
    lines += [plot_3_types(logits, targets, axs)]
    return axs, lines

  def tune(self, **kwargs):
    builder = self.model_class(n_classes=2, device=self.args.device)
    base_params = builder.get_parameters(task=self.args.task) | self.args.config | kwargs
    gtorch.train.tune.main(
      self.train_loader, self.val_loader,
      builder=builder, base_params=base_params,
      task=self.args.task, disk=self.args.disk)

  def find_lr(self, axs=None, params=None, label=None):
    builder = self.model_class(n_classes=2, device=self.args.device)
    base_params = builder.get_parameters(task=self.args.task) | params
    lrs, losses, conds = gtorch.train.lr_finder.find_lr(
        base_params, model_factory_fn=builder,
        train_loader=self.train_loader,
        task=self.args.task,
        disk=self.args.disk)
    losses, conds = gtorch.train.lr_plots.plot_lr(lrs, losses, conds=conds, smooth=len(self.train_loader), label=label, axs=axs)
    return losses, conds

  def get_lr_params(self, params=None):
    return dict(
        schedule="ramp",
        min_lr=1e-8,
        max_lr=1e+8,
        max_epochs=50,
    ) | self.args.config | (params or {})

  def find_momentum(self, momentum, params=None):
    assert momentum is not None
    params = self.get_lr_params(params)
    axs = gtorch.train.lr_plots.get_axes(params)
    loss, cond = zip(*[
      self.find_lr(axs, params=params | {"momentum": m}, label=f"momentum={m}")
      for m in momentum
    ])
    gtorch.train.lr_plots.show_axes(axs, loss, cond)