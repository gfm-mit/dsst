import random
from lxml import etree
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from etl.parse_semantics import *
from etl.parse_dynamics import *

PATH = "CIN0750296502.csv"

def plot_data(g):
  fig, axs = plt.subplots(2, 2)
  axs = axs.flatten()
  plt.sca(axs[0])
  plt.plot(g.x, g.y, color="lightgray")
  plt.title('v_mag2')
  plt.scatter(g.x, g.y, s=100, c=g.v_mag2 - 2.5, vmin=-4, vmax=4, cmap='BrBG')
  plt.sca(axs[1])
  plt.plot(g.x, g.y, color="lightgray")
  plt.title("dv_mag2")
  plt.scatter(g.x, g.y, s=100, c=g.dv_mag2, cmap='coolwarm', vmin=-4, vmax=4)
  plt.sca(axs[2])
  plt.plot(g.x, g.y, color="lightgray")
  plt.title("cw")
  plt.scatter(g.x, g.y, s=100, c=g.cw, cmap='PiYG', vmin=-4, vmax=4)
  plt.sca(axs[3])
  plt.plot(g.x, g.y, color="lightgray")
  plt.title('j_mag2')
  plt.scatter(g.x, g.y, s=100, c=g.j_mag2 - 2.5, vmin=-4, vmax=4, cmap='BrBG')
  plt.show()

if __name__ == "__main__":
  assigned_path = pathlib.Path("/Users/abe/Desktop/ASSIGNED/")
  dynamics_path = pathlib.Path("/Users/abe/Desktop/DYNAMICS/")
  np_path = pathlib.Path("/Users/abe/Desktop/NP/")
  dynamics_path.mkdir(parents=True, exist_ok=True)
  np_path.mkdir(parents=True, exist_ok=True)
  files = list(assigned_path.glob("*.csv"))
  random.shuffle(files)
  semantics = load_semantics()
  for csv in files:
    df = pd.read_csv(csv)
    df = assign_semantics(df, semantics)
    df = convert(df)
    df.to_csv(str(csv).replace("/ASSIGNED/", "/DYNAMICS/"))
    df2 = df["symbol task box t v_mag2 a_mag2 dv_mag2 cw j_mag2".split()]
    np.save(str(csv).replace("/ASSIGNED/", "/NP/").replace(".csv", ".npy"), df2.values)