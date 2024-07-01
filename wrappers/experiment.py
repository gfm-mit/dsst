import json
import wrappers.lr_finder
import plot.lr_finder
import core.train
import wrappers.tune
import core.metrics
import plot.calibration
import plot.metrics
import models.registry

class Experiment:
  def __init__(self, model_class, train_loader, val_loader, calibration_loader, test_loader, args):
    self.model_class = model_class
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.calibration_loader = calibration_loader
    self.test_loader = test_loader
    self.args = args
    self.model = None
    self.last_train_params = None
    self.last_metric = None
    self.n_classes = 2
    if args.task == "classify_section":
      self.n_classes = 6

  def train(self, tqdm_prefix=None, **kwargs):
    #torch.manual_seed(42)
    self.model_class = models.registry.lookup_model(self.args.model or kwargs.get("model", "linear"))
    builder = self.model_class(n_classes=self.n_classes, device=self.args.device)
    base_params = builder.get_parameters(task=self.args.task) | self.args.config | kwargs
    metric, epoch_loss_history, self.model = core.train.setup_training_run(
        base_params, model_factory_fn=builder,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        task=self.args.task,
        disk=self.args.disk,
        use_loss_history=self.args.history == "loss",
        tqdm_prefix=tqdm_prefix,
        offset=self.args.offset)
    if self.args.log != "":
      self.log_params = base_params
    return metric, epoch_loss_history

  def log_training(self, history, label):
    log_content = dict(
      args=vars(self.args),
      model_class=self.model_class.__name__,
      params=self.log_params,
      epoch_loss_history=history,
    )
    with open(self.args.log + "/{label}.json".format(label=label), "w") as f:
      json.dump(log_content, f, indent=2)
  
  def batch_eval_test(self):
    self.model.eval()
    logits, targets = core.metrics.get_combined_roc(
      self.model, self.test_loader,
      calibration_loader=self.calibration_loader,
      combine_fn=None if self.args.task == "classify" else core.metrics.linear_combiner)
    return logits, targets

  def plot_trained(self, axs, label=None):
    assert self.model is not None
    assert self.args.task in 'classify classify_patient classify_section'.split()
    logits, targets = self.batch_eval_test()
    roc = plot.calibration.get_full_roc_table(logits, targets)
    axs = plot.metrics.plot_palette(roc, axs, label=label)
    return axs

  def tune(self, **kwargs):
    builder = self.model_class(n_classes=self.n_classes, device=self.args.device)
    base_params = builder.get_parameters(task=self.args.task) | self.args.config | kwargs
    wrappers.tune.main(
      self.train_loader, self.val_loader,
      builder=builder, base_params=base_params,
      task=self.args.task, disk=self.args.disk, history=self.args.history,
      tuning_ranges=self.args.tune)

  def find_lr(self, axs=None, params=None, label=None):
    builder = self.model_class(n_classes=self.n_classes, device=self.args.device)
    base_params = builder.get_parameters(task=self.args.task) | params
    lrs, losses, conds = wrappers.lr_finder.find_lr(
        base_params,
        model_factory_fn=builder,
        train_loader=self.train_loader,
        task=self.args.task,
        disk=self.args.disk,
        tqdm_desc=label)
    losses, conds = plot.lr_finder.plot_lr(lrs, losses, conds=conds, smooth=len(self.train_loader), label=label, axs=axs)
    return losses, conds

  def get_lr_params(self, params=None):
    return dict(
        scheduler="ramp",
        min_lr=1e-5,
        max_lr=1,
        max_epochs=50,
    ) | self.args.find_lr | (params or {})

  def find_momentum(self, momentum, params=None):
    assert momentum is not None
    params = self.get_lr_params(params)
    axs = plot.lr_finder.get_axes(params)
    loss, cond = zip(*[
      self.find_lr(axs, params=params | {"momentum": m}, label=f"momentum={m}")
      for m in momentum
    ])
    plot.lr_finder.show_axes(axs, loss, cond)