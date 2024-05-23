import argparse
import sys

from plot.palette import plot_3_types, get_3_axes, draw_3_legends
import gtorch.datasets.synthetic
import gtorch.datasets.linear_patient
import gtorch.datasets.linear_box
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.optimize.metrics
import gtorch.hyper.params
import gtorch.hyper.tune
import gtorch.hyper.coef
import util.excepthook

if __name__ == "__main__":
  # Set the custom excepthook
  sys.excepthook = util.excepthook.custom_excepthook

  parser = argparse.ArgumentParser(description='Run a linear pytorch model')
  parser.add_argument('--coef', action='store_true', help='Plot coefficients')
  parser.add_argument('--tune', action='store_true', help='Tune parameters')
  parser.add_argument('--compare', action='store_true', help='Run on both linear_agg and linear')
  parser.add_argument('--device', default='cpu', help='torch device')
  args = parser.parse_args()

  axs = None
  lines = []
  train_loader, val_loader, test_loader = gtorch.datasets.linear_patient.get_loaders()
  if args.coef:
    # check coefficients
    gtorch.hyper.coef.get_coef_dist(
      builder=gtorch.models.linear.Linear(n_classes=2, device=args.device),
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader)
  elif args.tune:
    # tune parameters
    axs, line1 = gtorch.hyper.tune.main(train_loader, val_loader, test_loader, axs=axs, device=args.device, builder=gtorch.models.linear.Linear(n_classes=2, device=args.device))
  else:
    builder = gtorch.models.linear.Linear(n_classes=2, device=args.device)
    #torch.manual_seed(42)
    base_params = builder.get_parameters()
    #with cProfile.Profile() as pr:
    if True:  # just a placeholder for indentation
      retval, model = gtorch.hyper.params.setup_training_run(base_params, model_factory_fn=builder,
                                                             train_loader=train_loader, val_loader=val_loader)
    #pr.dump_stats('results/output_file.prof')
    model.eval()
    logits, targets = gtorch.optimize.metrics.get_combined_roc(model, test_loader, combine_fn=None)
    axs = get_3_axes() if axs is None else axs
    lines += [plot_3_types(logits, targets, axs)]
    if args.compare:
      train_loader, val_loader, test_loader = gtorch.datasets.linear_box.get_loaders()
      builder = gtorch.models.linear.Linear(n_classes=2, device=args.device)
      #torch.manual_seed(42)
      base_params = builder.get_parameters()
      base_params["max_epochs"] *= 100  # because less data
      retval, model = gtorch.hyper.params.setup_training_run(base_params, model_factory_fn=builder,
                                                             train_loader=train_loader, val_loader=val_loader)

      model.eval()
      logits, targets = gtorch.optimize.metrics.get_combined_roc(model, test_loader, combine_fn=gtorch.datasets.linear_box.combiner)
      axs = get_3_axes() if axs is None else axs
      lines += [plot_3_types(logits, targets, axs)]
    draw_3_legends(axs, lines)