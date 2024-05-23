import random
from lxml import etree
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from etl.parse_boxes import get_boxes

PATH = "CIN1678324519.csv"

if __name__ == "__main__":
  data_path = pathlib.Path("/Users/abe/Desktop/OUT/")
  assigned_path = pathlib.Path("/Users/abe/Desktop/ASSIGNED/")
  #debug = []
  files = list(data_path.glob("*.csv"))
  random.shuffle(files)
  for csv in files:
    df = pd.read_csv(csv)
    df = get_boxes(df, csv)
    out = str(csv).replace("/OUT/", "/ASSIGNED/")
    df.to_csv(out)
    #debug += [df]
  #debug = pd.concat(debug).fillna(0)