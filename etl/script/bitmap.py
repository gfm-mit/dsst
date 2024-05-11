import random
from lxml import etree
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from skimage.draw import line
from scipy.sparse import csr_matrix, save_npz, vstack
import re

from etl.parse_semantics import *
from etl.parse_dynamics import *

def rasterize_path_to_bitmap(points, bitmap=None):
  for i in range(len(points) - 1):
      x1, y1 = points[i]
      x2, y2 = points[i + 1]
      rr, cc = line(y1, x1, y2, x2)
      good = (rr < bitmap.shape[0]) & (cc < bitmap.shape[1])
      rr, cc = rr[good], cc[good]
      bitmap[rr, cc] = 1
  return bitmap

if __name__ == "__main__":
  assigned_path = pathlib.Path("/Users/abe/Desktop/ASSIGNED/")
  bitmap_path = pathlib.Path("/Users/abe/Desktop/BITMAP/")
  bitmap_path.mkdir(parents=True, exist_ok=True)
  files = list(assigned_path.glob("*.csv"))
  random.shuffle(files)
  semantics = load_semantics()
  W = 64
  for csv in files:
    pkey = next(re.finditer(r"/(CIN\d+).csv$", str(csv))).group(1)
    df = pd.read_csv(csv)
    #print("df.head().T", df.head().T)
    df = assign_semantics(df, semantics)
    meta = []
    data = []
    for (symbol, task, box), g in df.groupby("symbol task box".split()):
      bitmap = np.zeros([W, W], dtype=np.uint8)
      for _, stroke in g.groupby("stroke_id"):
        xy = stroke["x y".split()].copy()
        xy.x += .1
        xy.y -= .1
        xy /= 1.2
        xy = xy[xy.x.between(0, 1)]
        xy = xy[xy.y.between(0, 1)]
        xy.y = 1 - xy.y
        xy = xy.values
        xy = np.round(W * xy).astype(int)
        rasterize_path_to_bitmap(xy, bitmap)
      if bitmap.sum() == 0:
        print("empty box", pkey, symbol, task, box)
      bitmap = csr_matrix(bitmap.reshape([-1]))
      meta += [(symbol, task, box)]
      data += [bitmap]
    data = vstack(data)
    meta = pd.DataFrame(meta, columns="symbol task box".split())
    meta.to_csv(bitmap_path / f"{pkey}.csv")
    save_npz(bitmap_path / f"{pkey}.npz", data)
    print(pkey)