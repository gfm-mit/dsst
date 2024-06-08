import gtorch.datasets.linear_box
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
    base_params = builder.get_parameters(pretraining=self.args.pretraining) | kwargs
    retval, self.model = gtorch.hyper.params.setup_training_run(
        base_params, model_factory_fn=builder,
        train_loader=self.train_loader,
        val_loader=self.val_loader,
        pretraining=self.args.pretraining)
    return retval

  def plot_trained(self, axs, lines):
    assert self.model is not None
    self.model.eval()
    logits, targets = gtorch.optimize.metrics.get_combined_roc(
      self.model, self.test_loader,
      combine_fn=None if self.args.box_level else gtorch.datasets.linear_box.combiner)
    axs = get_3_axes() if axs is None else axs
    lines = [] if lines is None else lines
    lines += [plot_3_types(logits, targets, axs)]
    return axs, lines

  def tune(self):
    builder = self.model_class(n_classes=2, device=self.args.device)
    gtorch.hyper.tune.main(
      self.train_loader, self.val_loader,
      builder=builder, pretraining=self.args.pretraining)