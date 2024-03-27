import random
from lxml import etree
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from etl.parse_xml import *
from etl.parse_boxes import *

PATH = "CIN1314662258-2016-11-02-15-42-25V6.5.ssk"

if __name__ == "__main__":
  data_path = pathlib.Path("/Users/abe/Desktop/OUT/")
  coords = []
  files = list(data_path.glob("*.csv"))
  random.shuffle(files)
  files = """
/Users/abe/Desktop/OUT/CIN1846253767.csv
/Users/abe/Desktop/OUT/CIN0646790786.csv
/Users/abe/Desktop/OUT/CIN0743785657.csv
/Users/abe/Desktop/OUT/CIN0428172784.csv
/Users/abe/Desktop/OUT/CIN1045085968.csv

/Users/abe/Desktop/OUT/CIN1678324519.csv
/Users/abe/Desktop/OUT/CIN1349417759.csv
/Users/abe/Desktop/OUT/CIN0211074041.csv
/Users/abe/Desktop/OUT/CIN0628101242.csv
/Users/abe/Desktop/OUT/CIN1622623454.csv
/Users/abe/Desktop/OUT/CIN1704088530.csv
/Users/abe/Desktop/OUT/CIN0783760789.csv
/Users/abe/Desktop/OUT/CIN1380842865.csv
/Users/abe/Desktop/OUT/CIN1708459300.csv
/Users/abe/Desktop/OUT/CIN0452502449.csv
/Users/abe/Desktop/OUT/CIN0916103729.csv
/Users/abe/Desktop/OUT/CIN0449422271.csv
  """.split()
  files = [f for f in files if f]
  for csv in files:
    df = pd.read_csv(csv)
    df = get_boxes(df, csv)
    #df.x = df.x / 12.7 - 2.1
    #df.y = df.y / 12.7 - 4
    coords += [df]
  exit(0)
  coords = pd.concat(coords).fillna(0)
  coords.to_csv("junk.csv")
  coords = pd.read_csv("junk.csv")
  #print(coords.head())
  #plt.hist(coords.loc[:, "x_back"], bins=100, log=True)
  #plt.hist(coords.loc[:, "x_forwards"], bins=100, log=True)
  plt.show()
  exit(0)

  #m, bx, by, dy = guess_width(coords.x, coords.y, 12.73)
  #print(m, bx, by, dy)
  #m, bx, by = 12.73, 2, 4.1
  #print(coords.head())
  ##for g, df in coords.groupby("stroke_id"):
  #coords.groupby("stroke_id")
  df["x_int"] = np.round(df.x)
  delta = pd.DataFrame(np.diff(df, axis=0), columns=df.columns)
  delta = delta[delta.stroke_id != 0]
  delta["x1"] = np.roll(df.loc[delta.index].x.values, -1)
  delta["x1"].iloc[-1] = np.nan
  delta["c"] = np.select(
    [
      delta.x.between(-.5, .5) & delta.y.between(-.75, .25),# & delta.x_int.equals(0),
      delta.x.between(.5, 1.5) & delta.y.between(-.75, .75),# & delta.x_int.equals(1),
      delta.x.between(-15, -5) & delta.x1.between(-.5, .5),
      # & (delta.y.between(1, 2.5) | delta.y.between(4, 6)),
      True
    ],
    [0, 1, 2, 3]
  )
  #plt.scatter(delta.x, -delta.y, c=delta.c, cmap="tab10")
  #plt.scatter(delta.x1, -delta.y, c=delta.c, cmap="tab10")
  #delta.x_int = delta.x_int != 0
  fig, axs = plt.subplots(1, 2, figsize=(30, 10))
  plt.sca(axs[0])
  plt.plot(df.x, -df.y, color="lightgray", zorder=-20)
  for idx, g in df.groupby("stroke_id"):
    color = "black"
    if g.x.round().nunique() > 1:
      color = "orange"
    plt.plot(g.x, -g.y, color=color, zorder=-10)
  df["c"] = np.nan
  df.c[delta.index] = delta.c
  df2 = df[df.c.notna()]
  df2 = df2[~df2.c.isin([0, 1])]
  plt.scatter(df2.x, -df2.y, alpha=0.7, c=df2.c, cmap="tab10")
  #plt.xlim([-15, -10])
  #plt.ylim([-4, 0])
  plt.grid(True)
  plt.sca(axs[1])
  plt.hist(df.x % 1, bins=30)
  plt.show()