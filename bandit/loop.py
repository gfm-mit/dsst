import time
import matplotlib.pyplot as plt
import numpy as np

import bandit.base
import plot.tune

def run(bandit: bandit.base.Bandit, experiment):
  i = 0
  try:
    while (arm := bandit.suggest_arm()) is not None:
      i += 1
      label = bandit.get_label()
      tqdm_prefix=f"Tuning[{i}/{bandit.conf['budget']}]={label}"
      #if 'batch' in arm:
      #  experiment.redefine_loaders(arm['batch'])
      if 'columns' in arm:
        experiment.redefine_loaders(batch_size=1000, trunc=arm['trunc'])
      delay = time.time()
      metric, epoch_loss_history = experiment.train(tqdm_prefix=tqdm_prefix, **arm)
      metric = experiment.get_tuning_results(epoch_loss_history)
      metric["delay"] = time.time() - delay
      bandit.update_rewards(metric)
      yield metric, epoch_loss_history, label
  except KeyboardInterrupt:
    print("KeyboardInterrupt")