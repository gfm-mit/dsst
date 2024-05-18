import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from plot.palette import plot_3_types, get_3_axes, draw_3_legends
import gtorch.datasets.synthetic
import gtorch.datasets.linear_agg
import gtorch.datasets.linear
import gtorch.models.linear
import gtorch.optimize.optimize
import gtorch.hyper.params
import gtorch.hyper.tune
import util.excepthook
import sys

def main(train_loader, val_loader, test_loader, axs=None, device='cpu'):
  #torch.manual_seed(42)
  model, base_params = gtorch.models.linear.get_model(hidden_width=2, device=device, classes=2)
  retval, model = gtorch.hyper.params.many_hyperparams(base_params, model_factory_fn=gtorch.models.linear.get_model,
                                                       train_loader=train_loader, val_loader=val_loader)
  results = []
  model.eval()
  return model

def get_coef(model):
  for k, v in model.state_dict().items():
    print(k, v)
  return pd.Series(
    np.concatenate([
      model.state_dict()['1.bias'].numpy(),
      model.state_dict()['1.weight'].numpy()[0]
    ])
  )

def get_roc(model, device='cpu'):
  with torch.no_grad():
    for idx, (data, target, g) in enumerate(test_loader):
      #print(target)
      output = model(data.to(device)).to('cpu')
      results += [pd.DataFrame(dict(
          logits=output.detach().numpy()[:, 1],
          targets=target.detach().to('cpu').numpy()[:, 0],
          groups=g,
      ))]
      if idx % 100 == 0:
        print(idx)
  # TODO: why is this thing not working at all?
  df = pd.concat(results)
  df = df.groupby("groups").mean()
  logits, targets = df.logits.values, df.targets.values

  axs = get_3_axes() if axs is None else axs
  line1 = plot_3_types(logits, targets, axs)
  return axs, line1
  draw_3_legends(axs, [line1])

def get_coef_dist(train_loader, val_loader, test_loader, axs=None, device='cpu'):
  results = []
  for _ in range(10):
    results += [get_coef(main(train_loader, val_loader, test_loader, axs=axs, device=device))]
  results = pd.DataFrame(results)
  print(results)
  map = []
  for e, col in enumerate(results.columns):
    print(col)
    plt.scatter(np.random.normal(loc=e, scale=0.1, size=results.shape[0]), results.loc[:, col], label="col", alpha=0.5)
  #artist = plt.scatter(np.arange(7), [-2.64393084, 1.49190833, -0.79180225, 0.57461165, 0.18255898, 0.42163847, -0.13575019], color='lightgray', zorder=-10, s=200)
  artist = plt.scatter(np.arange(7), [0, 1, 1, 1, -1, -1, -1], color='lightgray', zorder=-10, s=200)

  plt.axhline(y=0, color="lightgray", linestyle=':', zorder=-10)
  plt.xticks(*zip(*enumerate(results.columns)))
  plt.legend([artist], "sklearn".split())
  plt.title('PyTorch parameters vs SKLearn parameters')
  plt.show()
  print(results)

if __name__ == "__main__":
  # Set the custom excepthook
  sys.excepthook = util.excepthook.custom_excepthook
  axs = None
  train_loader, val_loader, test_loader = gtorch.datasets.synthetic.get_loaders()
  get_coef_dist(train_loader, val_loader, test_loader, axs=axs, device='cpu')
  # TODO: evil!  val and test on train set

  #axs, line1 = gtorch.hyper.tune.main(train_loader, val_loader, test_loader, axs=axs, device='cpu')
  #draw_3_legends(axs, [line1])

  #train_loader, val_loader, test_loader = gtorch.datasets.linear.get_loaders()
  #axs, line2 = main(train_loader, val_loader, test_loader, axs=axs)
  #draw_3_legends(axs, [line1, line2])
  #cProfile.run('main()', 'output_file.prof')