# flake8: noqa
import argparse
import pathlib
import pprint
import re
from einops import rearrange
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.gaussian_process
import sklearn.linear_model
import tomli
import plot.tune

rewards = pd.read_csv('results/bandit/rewards.csv', index_col=0)
arms = pd.read_csv('results/bandit/arms.csv', index_col=0)

error = rewards.join(arms, on="arm_idx").groupby("columns").auc.agg(lambda x: x.std() / x.count()).sort_values().mean() * 100
print(f"{error:.2f}%")

rewards = rewards.join(arms, on="arm_idx").groupby("columns").auc.mean().sort_values() * 100
ZERO = rewards.loc["0"]
V = rewards.loc["v_mag2"]
ALL = rewards.loc["t v_mag2 a_mag2 dv_mag2 cw j_mag2"]

stuff = pd.DataFrame(
  [{k: 1 for k in x.split()} for x in rewards.index],
  index=rewards.index
  ).fillna(0)
stuff = stuff.drop(columns=["0"])
stuff["baseline"] = stuff.sum(axis=1)
stuff["auc"] = rewards.values
stuff = stuff["baseline auc".split()].reset_index()
stuff = stuff.set_index("baseline columns".split())

ONE = stuff.loc[1] - ZERO
ONE.columns = ["ONE"]
TWO = stuff.loc[2] - V
TWO.index = TWO.index.str.replace("v_mag2 ", "")
TWO.columns = ["TWO"]
FIVE = ALL - stuff.loc[5]
FIVE.index = FIVE.index.map(
  lambda x: list(set("t v_mag2 a_mag2 dv_mag2 cw j_mag2".split()) - set(x.split()))[0]
)
FIVE.columns = ["FIVE"]
join = pd.concat([ONE, TWO, FIVE], axis=1)
join.loc["base"] = [ALL - ZERO, np.nan, np.nan]
print(join.round(1))