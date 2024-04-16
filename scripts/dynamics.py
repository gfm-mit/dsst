import random
from lxml import etree
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from etl.parse_semantics import *
from etl.parse_dynamics import *

PATH = "CIN0750296502.csv"

if __name__ == "__main__":
  assigned_path = pathlib.Path("/Users/abe/Desktop/ASSIGNED/")
  dynamics_path = pathlib.Path("/Users/abe/Desktop/DYNAMICS/")
  files = list(assigned_path.glob("*.csv"))
  random.shuffle(files)
  semantics = load_semantics()
  debug = []
  files = "/Users/abe/Desktop/ASSIGNED/CIN0750296502.csv".split()
  for csv in files:
    df = pd.read_csv(csv)
    df = assign_semantics(df, semantics)
    df = convert(df)
    #out = str(csv).replace("/OUT/", "/ASSIGNED/")
    #df.to_csv(out)
    #plt.plot(df.x, df.y, color="lightgray")
    for idx, g in df.groupby("box"):
      if idx <= 1:
        continue
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
    exit(0)
    #debug += [df.groupby("task box".split()).t.min()]
  #print(debug)