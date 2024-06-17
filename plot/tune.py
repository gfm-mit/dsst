import matplotlib.pyplot as plt
import numpy as np


def plot_tuning_history(spaces, results, history):
    for e, row in results.iterrows():
      label = str({
        k: "{:.2E}".format(row[k]) if isinstance(row[k], float) else row[k]
        for k in spaces.keys()
      })
      plt.plot(row['history'], label=label)
    plt.legend()
    plt.axhline(y=.725, color="gray", zorder=-10)
    plt.xlabel('epoch')
    plt.ylabel(f'{history=}')
    #y_min = results.history.apply(lambda x: x[0]).min()
    #y_max = results.history.apply(max).max()
    y_min, y_max = .71, .725
    plt.ylim([y_min, y_max])
    plt.show()

def plot_tuning_results(spaces, results, task):
  N = int(np.ceil(np.sqrt(spaces.shape[1])))
  fig, axs = plt.subplots(N, N)
  if not isinstance(axs, np.ndarray):
    axs = [axs]
  else:
    axs = axs.flatten()
  metric_name = "MSE" if task == "next_token" else "AUC-ROC-C"
  for e, k in enumerate(spaces.columns):
    plt.sca(axs[e])
    plt.scatter(results[k], results['metric'])
    plt.ylabel(metric_name)
    plt.xlabel(k)
    if results[k].dtype == object:
      plt.xticks(rotation=45)
    elif 0 < results[k].min() < results[k].max() < 1:
      plt.xscale('logit')
    elif 0 < results[k].min() < 1:
      plt.xscale('log')
    else:
      pass
  plt.suptitle("Aggregated at Box Level, not Patient Level")
  plt.show()
  print(results)
  return axs, None
