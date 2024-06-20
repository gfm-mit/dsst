import argparse
import sys

#from hashlib import sha1
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

import etl.torch.linear_box
import etl.torch.linear_patient
import etl.torch.synthetic
import util.excepthook
import plot.metrics
import plot.calibration


def main(train_loader, val_loader, test_loader, axs=None, random_state=None, solver=None, combine_fn=None):
  x, y = [z[0] for z in zip(*train_loader)]
  x = x[:, :, 0]
  y = y[:, 0]

  #hash = int(sha1(bytes(str(y), 'utf8')).hexdigest(), 16) & ((1 << 16) - 1)
  #print(f"{hash:0b}", y[:4])
  model = LogisticRegression(random_state=random_state, solver=solver, max_iter=1 << 9, multi_class='multinomial')
  model.fit(x, y)

  #return measure_conditioining(x, y, model)

  print("bias", model.intercept_, "coef", model.coef_)

  x, y, g = [z[0] for z in zip(*test_loader)]
  x = x[:, :, 0]
  y = y[:, 0]
  logits = model.predict_log_proba(x)[:, 1]
  targets = y.numpy()
  if combine_fn is not None:
    logits, targets = combine_fn(logits, targets, g)

  roc = plot.calibration.get_full_roc_table(logits, targets)
  axs = plot.metrics.plot_palette(roc, axs)

def measure_conditioning(x, y, model):
    print("cond", np.linalg.cond(x @ x.T))
    base = np.array(model.coef_)
    basis = 1e-2 * np.eye(base.shape[1])

    def score(delta):
      model.coef_ = base + delta
      score = model.score(x, y)
      model.coef_ = base
      return score
    delta1 = np.zeros(basis.shape[0])
    delta2 = np.zeros_like(basis)
    zero = score(0 * basis[0])
    curvature = np.zeros_like(basis)
    for i in range(basis.shape[0]):
      delta1[i] = score(basis[i]) - zero
    for i in range(basis.shape[0]):
      for j in range(basis.shape[1]):
        delta2[i, j] = score(basis[i] + basis[j]) - zero
        curvature[i, j] = delta2[i, j] - delta1[i] - delta1[j]
    print("vals", np.linalg.eigvals(curvature))
    print("curvature", np.linalg.cond(curvature))
    curvature /= np.max(np.abs(curvature))
    plt.imshow(curvature, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()
    exit(0)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Use SKLearn on the pytorch data loader')
  parser.add_argument('--solver', default='lbfgs', help='SKLearn Logistic Regresion Solver')
  args = parser.parse_args()
  axs = None
  lines = []
  sys.excepthook = util.excepthook.custom_excepthook
  train_loader, val_loader, test_loader = etl.torch.linear_box.get_loaders()
  main(train_loader, val_loader, test_loader, axs=axs, random_state=42, solver=args.solver, combine_fn=etl.torch.linear_box.combiner)
  plt.suptitle("linear_box.combiner")
  plt.tight_layout()
  plt.show()