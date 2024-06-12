import gtorch.datasets.linear_box
import gtorch.hyper.lr_finder
import gtorch.hyper.lr_plots
import gtorch.hyper.params
import gtorch.hyper.tune
import gtorch.optimize.metrics
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
    base_params = builder.get_parameters(task=self.args.task) | kwargs
    metric, epoch_loss_history, self.model = gtorch.hyper.params.setup_training_run(
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
    logits, targets = gtorch.optimize.metrics.get_combined_roc(
      self.model, self.test_loader,
      combine_fn=None if self.args.task == "classify" else gtorch.datasets.linear_box.combiner)
    axs = get_3_axes() if axs is None else axs
    lines = [] if lines is None else lines
    lines += [plot_3_types(logits, targets, axs)]
    return axs, lines

  def tune(self):
    builder = self.model_class(n_classes=2, device=self.args.device)
    gtorch.hyper.tune.main(
      self.train_loader, self.val_loader,
      builder=builder, task=self.args.task, disk=self.args.disk)

  def find_lr(self, axs=None, lr_params=None, label=None):
    builder = self.model_class(n_classes=2, device=self.args.device)
    lr_params = dict(
        schedule="ramp",
        min_lr=1e-5,
        max_lr=1e3,
        max_epochs=30,
    ) | (lr_params if lr_params is not None else {})
    base_params = builder.get_parameters(task=self.args.task) | lr_params
    lrs, losses, conds = gtorch.hyper.lr_finder.find_lr(
        base_params, model_factory_fn=builder,
        train_loader=self.train_loader,
        task=self.args.task,
        disk=self.args.disk)
    losses, conds = gtorch.hyper.lr_plots.plot_lr(lrs, losses, conds=conds, smooth=len(self.train_loader), label=label, axs=axs)
    return losses, conds

  def find_momentum(self, lr_params, momentum=None):
    assert momentum is not None
    axs = gtorch.hyper.lr_plots.get_axes(lr_params)
    loss, cond = zip(*[
      self.find_lr(axs, lr_params=lr_params | {"momentum": m}, label=f"momentum={m}")
      for m in momentum
    ])
    gtorch.hyper.lr_plots.show_axes(axs, loss, cond)