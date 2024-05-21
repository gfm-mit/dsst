import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import gtorch.optimize.metrics

def get_coef_dist(builder, train_loader, val_loader, test_loader):
  results = []
  for _ in range(10):
    #torch.manual_seed(42)
    base_params = builder.get_parameters()
    metrics, model = gtorch.hyper.params.setup_training_run(base_params, model_factory_fn=builder,
                                                         train_loader=train_loader, val_loader=val_loader)
    model.eval()
    coef = builder.get_coefficients(model)
    #coef = np.concatenate([[metrics], coef])
    results += [coef]
  results = pd.DataFrame(results)
  for e, col in enumerate(results.columns):
    print(col)
    plt.scatter(np.random.normal(loc=e, scale=0.1, size=results.shape[0]), results.loc[:, col], label="col", alpha=0.5)
  artist = plt.scatter(np.arange(13),
                      [-0.72, 0.05, 1.14, 0.03, 0.09, 0.19, -0.65, 0.09, -0.01, 0., -0.16, -0.20, -0.69], color='lightgray', zorder=-10, s=200)
  #artist = plt.scatter(np.arange(13), [0, 2, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -2], color='lightgray', zorder=-10, s=100)

  plt.axhline(y=0, color="lightgray", linestyle=':', zorder=-10)
  plt.xticks(*zip(*enumerate(results.columns)))
  plt.legend([artist], "sklearn".split())
  plt.title('PyTorch parameters vs SKLearn parameters')
  plt.ylim([-4, 4])
  plt.show()
  print(results)
