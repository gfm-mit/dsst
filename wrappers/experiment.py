import numpy as np
import scipy
import sklearn.metrics

import core.train
import core.metrics
import core.batch_eval
import plot.calibration
import plot.metrics
import models.registry

class Experiment:
  def __init__(self, model_class, loader_fn, train_loader, val_loader, calibration_loader, test_loader, args):
    self.model_class = model_class
    self.loader_fn = loader_fn
    self.args = args
    self.model = None
    self.last_train_params = None
    self.last_metric = None
    self.n_classes = 2
    if args.task == "classify_section":
      self.n_classes = 6
    self.batch_size = np.inf # sadly, this value != None
  
  def redefine_loaders(self, batch_size=None):
    if self.batch_size == batch_size:
      return
    print("redefining loaders: ", batch_size)
    self.train_loader, self.val_loader, self.calibration_loader, self.test_loader = self.loader_fn(
      device=self.args.device,
      task=self.args.task,
      batch_size=int(batch_size),
    )
    self.batch_size = int(batch_size)

  def train(self, tqdm_prefix=None, **kwargs):
    #torch.manual_seed(42)
    self.model_class = models.registry.lookup_model(self.args.model or kwargs.get("model", "linear"))
    builder = self.model_class(n_classes=self.n_classes, device=self.args.device)
    base_params = builder.get_parameters(task=self.args.task) | kwargs
    if "batch" not in base_params:
      self.redefine_loaders(1000) # bit arbitrary, but was used in original experiments
    if base_params["batch"] != self.batch_size:
      self.redefine_loaders(base_params["batch"])
    metric, epoch_loss_history, self.model = core.train.setup_training_run(
        base_params, model_factory_fn=builder,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        task=self.args.task,
        disk=self.args.disk,
        early_stopping=self.args.stats != "train_loss",
        tqdm_prefix=tqdm_prefix,
        offset=self.args.offset)
    return metric, epoch_loss_history
  
  def batch_eval_test(self):
    self.model.eval()
    if self.args.task == "next_token":
      logits, targets, groups = core.batch_eval.next_token(self.model, self.test_loader, offset=self.args.offset)
    else:
      logits, targets = core.metrics.get_combined_roc(
        self.model, self.test_loader,
        calibration_loader=self.calibration_loader,
        combine_fn=core.metrics.linear_combiner if self.args.task == "classify_patient" else None)
    return logits, targets

  def plot_trained(self, axs, label=None):
    assert self.model is not None
    assert self.args.task in 'classify classify_patient classify_section'.split()
    logits, targets = self.batch_eval_test()
    roc = plot.calibration.get_full_roc_table(logits, targets)
    axs = plot.metrics.plot_palette(roc, axs, label=label)
    return axs

  def get_tuning_results(self, epoch_loss_history):
    logits, targets = self.batch_eval_test()
    if self.args.task == "next_token":
      # dumb that sklearn RMSE doesn't work for 3 tensors
      rmse = np.sqrt(np.mean((targets - logits) ** 2))
      best_epoch = np.argmin(epoch_loss_history)
      return dict(rmse=rmse, best_epoch=best_epoch)
    else:
      plot_metric = sklearn.metrics.roc_auc_score(targets, logits)
      probs = scipy.special.expit(logits)
      brier = sklearn.metrics.brier_score_loss(targets, probs)
      best_epoch = np.argmax(epoch_loss_history)
      return dict(auc=plot_metric, brier=brier, best_epoch=best_epoch)