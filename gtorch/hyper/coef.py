import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_coef_1(model):
  for k, v in model.state_dict().items():
    print(k, v)
  return pd.Series(
    np.concatenate([
      model.state_dict()['1.bias'].numpy(),
      model.state_dict()['1.weight'].numpy().flatten(),
    ])
  )

def get_coef_2(model):
  for k, v in model.state_dict().items():
    print(k, v)
  return pd.Series(
    np.concatenate([
      [model.state_dict()['1.bias'].numpy()[1]
      - model.state_dict()['1.bias'].numpy()[0]],
      model.state_dict()['1.weight'].numpy()[1]
      - model.state_dict()['1.weight'].numpy()[0]
    ])
  )

def get_coef_dist(train_model):
  results = []
  for _ in range(10):
    results += [get_coef_1(train_model())]
  results = pd.DataFrame(results)
  print(results)
  map = []
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
