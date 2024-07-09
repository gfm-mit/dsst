import numpy as np

import bandit.base

class UCB(bandit.base.Bandit):
  def calculate_state(self): 
    # std n ucb
    counts = self.rewards.groupby("arm_idx").size().rename("n")
    state = counts.reindex(self.arms.index, fill_value=0).to_frame()
    state["mu"] = self.rewards.groupby("arm_idx")[self.metric].mean()
    state["sigma"] = self.rewards.groupby("arm_idx")[self.metric].std(ddof=1)
    state.sigma = state.sigma.fillna(state.sigma.max(skipna=True))
    print(f"{state=}")
    
    with np.errstate(divide='ignore', invalid='ignore'):
      ucb = 4 * np.log(state.n.sum() - 1) / state.n
      ucb = 2 * state.sigma * np.sqrt(ucb)
    ucb[state.n == 0] = np.inf
    ucb = ucb.fillna(-np.inf) # annoying edge case when N=1 overall, or there are no sigmas

    if self.task == "next_token":
      state.ucb = state.mu.fillna(-np.inf) - ucb
      best_idx = state.ucb.argmin()
    else:
      state.ucb = state.mu.fillna(np.inf) + ucb
      best_idx = state.ucb.argmax()
    state.to_csv("results/bandit/state.csv")
    return best_idx