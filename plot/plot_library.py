import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def get_log_weighted_average(x, y):
  return np.trapz(y, np.log(x)) / np.trapz(1 + 0 * y, np.log(x))

def plot_roc_normal(roc, convex, interpolated_smooth, axs, color):
  plt.title("ROC")
  color = plt.scatter(roc.fpr_literal, roc.tpr_literal, s=1).get_facecolor()[0]
  plt.scatter(convex.fpr[1:-1], convex.tpr[1:-1], s=100, alpha=0.2, color=color)
  plt.plot(convex.fpr, convex.tpr, linewidth=3, alpha=0.4, color=color)
  plt.ylabel('TPR')
  plt.yticks([0, .5, 1], "0% 50% 100%".split())
  plt.xlabel('FPR')
  plt.xticks([0, .5, 1], "0% 50% 100%".split())
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  return color

def plot_roc_swets(roc, convex, interpolated_smooth, axs, color):
  plt.title("ROC (Q-Q plot)")
  color = plt.scatter(roc.fpr_literal, roc.tpr_literal, s=1, alpha=0.7, color=color).get_facecolor()[0]
  artist = plt.plot(roc.fpr_literal, roc.tpr_literal, alpha=0.2, color=color)[0]
  #plt.scatter(convex.fpr[1:-1], convex.tpr[1:-1], s=100, alpha=0.2, color=color)
  plt.plot(interpolated_smooth.fpr, interpolated_smooth.tpr, linewidth=3, alpha=0.4, color=color)

  cz0 = 1e-3
  z0 = norm.ppf(cz0)
  cz1 = 1-1e-3
  z1 = norm.ppf(cz1)

  plt.ylabel("TPR")
  plt.yscale('probit')
  plt.ylim([cz0, cz1])
  plt.yticks([1e-2, 1e-1, 5e-1, 1-1e-1, 1-1e-2], "1% 10% 50% 90% 99%".split())

  plt.xlabel("FPR")
  plt.xscale('probit')
  plt.xlim([cz0, cz1])
  plt.xticks([1e-2, 1e-1, 5e-1, 1-1e-1, 1-1e-2], "1% 10% 50% 90% 99%".split())
  #plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()

  plt.axline((cz0, cz0), (cz1, cz1), color="lightgray", linestyle = "--", zorder=-10)
  plt.axline((cz0, norm.cdf(z0+1)), (norm.cdf(z1-1), cz1), color="lightgray", linestyle = "--", zorder=-10)
  plt.axline((cz0, norm.cdf(z0+2)), (norm.cdf(z1-2), cz1), color="lightgray", linestyle = "--", zorder=-10)
  plt.gca().set_aspect('equal')
  return artist, color

def plot_roc_affine(roc, convex, interpolated_smooth, axs, color):
  plt.title('ROC (affine transform)')
  plus = convex.tpr + convex.fpr
  minus = convex.tpr - convex.fpr
  plt.plot(-plus, 1+minus, linewidth=3, alpha=0.4, color=color)
  plt.scatter(-plus[1:-1], 1+minus[1:-1], linewidth=3, s=100, alpha=0.2, color=color)
  plus = roc.tpr_literal + roc.fpr_literal
  minus = roc.tpr_literal - roc.fpr_literal
  plt.scatter(-plus, 1+minus, s=1)

  plt.plot([-0, -1, -2], [1, 2, 1], linestyle='--', color="lightgray", zorder=-10)
  plt.plot([-0, -2], [1, 1], linestyle=':', color="lightgray", zorder=-10)
  plt.ylabel('TPR + TNR\n(TPR + 1 - FPR)')
  plt.yticks([1, 1.5, 2], "100% 150% 200%".split())
  plt.xlabel('TPR + FPR (backwards)')
  plt.xticks([-0, -1, -2], "0% 100% 200%".split())
  plt.gca().set_aspect('equal')
  return color

def plot_rotated_roc(roc, convex, color):
  plt.title('ROC (rotated)')
  plus = convex.tpr + convex.fpr
  minus = convex.tpr - convex.fpr
  plt.plot(plus, minus, linewidth=3, alpha=0.4, color=color)
  plt.scatter(plus[1:-1], minus[1:-1], linewidth=3, s=100, alpha=0.2, color=color)
  plus = roc.tpr_literal + roc.fpr_literal
  minus = roc.tpr_literal - roc.fpr_literal
  plt.scatter(plus, minus, s=1)

  plt.plot([0, 1, 2], [0, 1, 0], linestyle='--', color="lightgray", zorder=-10)
  plt.plot([0, 2], [0, 0], linestyle=':', color="lightgray", zorder=-10)
  plt.ylabel('TPR - FPR')
  plt.xlabel('TPR + FPR')
  plt.xticks([0, 1, 2], "0% 100% 200%".split())
  plt.yticks([0, .5, 1], "0% 50% 100%".split())
  plt.gca().set_aspect('equal')
  return color

def plot_eta_conjugate(eta_density, eta, idx, roc, convex, interpolated_smooth, axs, color):
  plt.title('Convex Conjugate')
  minus = convex.tpr[idx] - convex.fpr[idx]
  #plt.plot(eta, minus, linewidth=10, alpha=0.2, color=color)
  minus[np.roll(minus, 1) == np.roll(minus, -1)] = np.nan
  plt.plot(eta, 1+minus, linewidth=2, alpha=0.4, color=color)
  minus = convex.tpr[idx] - convex.fpr[idx]
  minus[np.roll(minus, 1) != np.roll(minus, -1)] = np.nan
  plt.plot(eta, 1+minus, linewidth=10, alpha=0.2, color=color)
  plt.plot(eta, 1+np.ones_like(eta), linestyle='--', color="lightgray", zorder=-10)
  plt.plot(eta, 1+np.zeros_like(eta), linestyle=':', color="lightgray", zorder=-10)
  plt.xlabel('η (L.R. Positive Threshold)')
  plt.xscale('log')
  plt.xticks([.1, 1, 10], "0.1x 1x 10x".split())
  plt.ylabel('TPR + TNR')
  plt.yticks([1, 1.5, 2], "100% 150% 200%".split())
  return color

def plot_eta_accuracy(eta_density, eta, idx, roc, convex, interpolated_smooth, axs, color):
  plt.title('Balanced Accuracy\n(equivalent)')
  tpr_equiv = convex.tpr[idx] - eta * convex.fpr[idx]
  tnr_equiv = 1 - convex.fpr[idx] - (1-convex.tpr[idx]) / eta
  balanced = 1 - (1-tpr_equiv) / (1+eta)
  plt.plot(eta, balanced)
  neutral = np.where(eta < 1, 1 / (1+eta), 1 - 1 / (1+eta))
  plt.plot(eta, np.ones_like(eta), color="lightgray", linestyle='--', zorder=-10)
  plt.plot(eta, neutral, color="lightgray", linestyle=':', zorder=-10)
  plt.xlabel('η (L.R. Needed for Positive)')
  plt.xscale('log')
  plt.xticks([.1, 1, 10], "0.1x 1x 10x".split())
  plt.ylabel('Percent of Value')
  plt.yticks([0, .25, .5, 0.75, 1], "Guessing 25% 50% 75% Oracle".split())

def plot_eta_tpr_tnr(eta_density, eta, idx, roc, convex, interpolated_smooth, axs, color):
  plt.title('pure TPR or TNR\n(equivalent)')
  flat_idx = tpr_equiv == np.roll(tpr_equiv, 1)
  plt.plot(eta[flat_idx], tpr_equiv[flat_idx], linestyle='--', color=color)
  plt.plot(eta[~flat_idx], tpr_equiv[~flat_idx], color=color)
  flat_idx = tnr_equiv == np.roll(tnr_equiv, -1)
  plt.plot(eta[flat_idx], tnr_equiv[flat_idx], linestyle='--', color=color, alpha=0.4)
  plt.plot(eta[~flat_idx], tnr_equiv[~flat_idx], color=color, alpha=0.4)
  plt.xlabel('Likelihood Ratio')
  plt.xscale('log')
  plt.ylabel('TPR')
  plt.ylim([0, 1])
  ax2 = plt.gca().twinx()
  ax2.set_ylabel('TNR', color="lightgray")
  ax2.tick_params(axis='y', color="lightgray", labelcolor="lightgray")
  return color

def plot_eta_density(eta_density, eta, idx, roc, convex, interpolated_smooth, axs, color):
  window = (1e-1 < eta) & (eta < 1e1)
  plt.plot(eta[window], eta_density[window], color=color, alpha=0.5)
  plt.ylabel('Density')
  plt.xlabel('Likelihood Ratio')
  plt.xscale('log')
  plt.yscale('log')

def plot_eta_percent(eta_density, eta, idx, roc, convex, interpolated_smooth, axs, color, eta_hat=None):
  plt.title('Surplus\n(Constant -> Oracle)')
  tpr_equiv = convex.tpr[idx] - eta * convex.fpr[idx]
  tnr_equiv = 1 - convex.fpr[idx] - (1-convex.tpr[idx]) / eta
  balanced = 1 - (1-tpr_equiv) / (1+eta)
  neutral = np.where(eta < 1, 1 / (1+eta), 1 - 1 / (1+eta))
  frac = 1 - (1 - balanced) / (1 - neutral)
  artist = plt.plot(eta, frac)[0]
  plt.xlabel(r'$\hat{Y}(x) = \mathbb{1}\left[\frac{\ell(x\mid 1)}{\ell(x\mid 0)} \geq \eta \right]$')
  plt.xscale('log')
  plt.xticks([.1, 1, 10], "0.1x 1x 10x".split())
  plt.ylabel(r'$\mathbb{E}\ \left[U(\hat{Y}(x), Y)\right]$')
  y = plt.gca().get_yticks()
  plt.yticks(y, [f"{int(100 * y)}%" for y in y])
  if eta_hat is not None:
    plt.axvline(x=eta_hat, color="lightgray", linestyle='--', zorder=-10)
  idx = np.abs(np.log10(eta)) < 1
  return color, artist, get_log_weighted_average(eta[idx], 100 * frac[idx])

def plot_precision_recall(eta, idx, roc, convex, interpolated_smooth, axs, color):
  plt.plot(interpolated_smooth.tpr, interpolated_smooth.tpr / (interpolated_smooth.tpr + interpolated_smooth.fpr), linewidth=3, alpha=0.2, color=color)
  plt.scatter(roc.tpr_literal, roc.tpr_literal / (roc.tpr_literal + roc.fpr_literal), s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('Recall')
  plt.xticks([0, .5, 1], "0% 50% 100%".split())
  plt.ylabel('Precision')
  plt.yticks([0, .5, 1], "0% 50% 100%".split())
  return color

def plot_precision_at_k(eta, idx, roc, convex, interpolated_smooth, axs, color):
  budget = interpolated_smooth.tpr * roc.labels.sum() + interpolated_smooth.fpr * (1-roc.labels).sum()
  precision = interpolated_smooth.tpr * roc.labels.sum() / budget
  budget = budget / roc.shape[0]
  artist = plt.plot(budget, precision, linewidth=3, alpha=0.2, color=color)[0]
  budget = roc.tpr_literal * roc.labels.sum() + roc.fpr_literal * (1-roc.labels).sum()
  precision = roc.tpr_literal * roc.labels.sum() / budget
  budget = budget / roc.shape[0]
  plt.scatter(budget, precision, s=1)
  plt.xlabel('K/N')
  plt.xscale('log')
  plt.xlim([3e-2, 1])
  plt.xticks([1e-2, 1e-1, 1], "1% 10% 100%".split())
  plt.ylabel('Precision')
  y = plt.gca().get_yticks()
  plt.yticks(y, [f"{int(100 * y)}%" for y in y])
  plt.axhline(y=roc.labels.sum() / roc.shape[0], color="lightgray", linestyle='--', zorder=-10)
  plt.title('Precision@K')

  return color, artist, get_log_weighted_average(budget, 100 * precision)

def plot_nne_at_k(eta, idx, roc, convex, interpolated_smooth, axs, color):
  budget = interpolated_smooth.tpr * roc.labels.sum() + interpolated_smooth.fpr * (1-roc.labels).sum()
  nne = budget / interpolated_smooth.tpr / roc.labels.sum()
  budget = budget / roc.shape[0]
  plt.plot(budget, nne, linewidth=3, alpha=0.2, color=color)
  budget = roc.tpr_literal * roc.labels.sum() + roc.fpr_literal * (1-roc.labels).sum()
  nne = budget / roc.tpr_literal / roc.labels.sum()
  budget = budget / roc.shape[0]
  plt.scatter(budget, nne, s=1)
  plt.gca().xaxis.set_label_position('top') 
  plt.gca().xaxis.tick_top()
  plt.xlabel('% Predicted Positive\n(Triage Target)')
  plt.xscale('log')
  plt.xlim([1e-2, 1])
  plt.xticks([1e-2, 1e-1, 1], "1% 10% 100%".split())
  plt.ylabel('NNE (PP / TP)')
  return color

def plot_supply_demand(eta, idx, roc, convex, interpolated_smooth, axs, color):
  if convex.shape[0] > 2:
    N = convex.idx.max()
    # TODO: figure out the outliers on both sides... laplace?
    tpr_smooth = (convex.tpr * N + 1) / (N + 2)
    fpr_smooth = (convex.fpr * N + 1) / (N + 2)
    tpr_smooth[0] = 1
    fpr_smooth[0] = 1
    tpr_smooth.iloc[-1] = 0
    fpr_smooth.iloc[-1] = 0
    slope = np.diff(tpr_smooth) / np.diff(fpr_smooth)
    slope[-1] = np.maximum(slope[-1], slope[-2])
    plt.stairs(slope, 1 - convex.idx / N, baseline=np.inf, color=color)
  else:
    plt.plot([0, 1], [1, 1], color=color)
  plt.xlabel('Q(uantity)')
  plt.ylabel('P(rice)')
  plt.yscale('log')
  plt.title('Demand')