import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

def rescale_and_cut(df):
  # these are magic numbers, deduced by trial and error
  # it absolutely can be done with scipy.optimize.fmin, but getting the objective right is hard
  df.x = df.x / 12.6 - 1.6
  df.y = df.y / 12.6 - 2.9 - (1-df.symbol_digit) * 10.8
  minmax = df.groupby("stroke_id").y.agg(['min', 'max'])
  bad_strokes = minmax[minmax["min"] < 0]
  df = df[~df.stroke_id.isin(bad_strokes.index)].copy()
  df.y *= -1
  return df

# this is the machinery to optimize numerically instead of by hand - it's not as good
def score_guess(x):
  return -np.sum(np.cos(x * 2 * np.pi))

def score_guess2(x, y, dy, y0=3, y1=11, y2=13):
  dy = (y > y2) * dy
  grid = np.cos(x * 2 * np.pi) + np.cos((y + dy) * 2 * np.pi)
  grid[y < y0] = 0
  grid[y.between(y1, y2)] = 0
  return -np.sum(grid)

def guess_width(x, y, m):
  def score_wrapper(args):
    m, bx, by, dy = args
    return score_guess2(x / m - bx, y / m - by, dy)
  m, bx, by, dy = scipy.optimize.fmin(score_wrapper, [m, 0, 0, 0])
  return m, bx, by, dy

# some functions to quantify the visual and temporal gap to previous and next strokes
def get_jumps(df):
  jumps = df.groupby("stroke_id").x.agg(
      this_min="min",
      this_max="max",
  )
  jumps["prev_min"] = jumps["this_min"].shift(1).values
  jumps["prev_max"] = jumps["this_max"].shift(1).values
  jumps["next_min"] = jumps["this_min"].shift(-1).values
  jumps["next_max"] = jumps["this_max"].shift(-1).values
  jumps["prev_min"].iloc[0] = np.nan
  jumps["prev_max"].iloc[0] = np.nan
  jumps["next_min"].iloc[-1] = np.nan
  jumps["next_max"].iloc[-1] = np.nan

  time = df.groupby("stroke_id").t.agg(['min', 'max'])
  jumps["t_prev"] = time["min"].values - time["max"].shift(1).values
  jumps["t_next"] = time["min"].shift(-1).values - time["max"].values
  jumps["t_prev"].iloc[0] = np.nan
  jumps["t_next"].iloc[-1] = np.nan

  vertical = df.groupby("stroke_id").y.median()
  jumps["y_prev"] = vertical.values - vertical.shift(1).values
  jumps["y_next"] = vertical.shift(-1).values - vertical.values
  jumps["y_prev"].iloc[0] = np.nan
  jumps["y_next"].iloc[-1] = np.nan
  return jumps

def ratio_factory(jumps):
  def get_ratio(df):
    j = jumps.loc[df.name]
    mid_x = np.ceil(df.x.iloc[0]).astype(int)
    box_middle_x = 0.5 * (np.max(df.x) + np.min(df.x)) - mid_x 
    z = np.floor(df.x) - mid_x
    w = np.abs(df.x - np.round(df.x)) < 0.07
    z = z[~w]

    width = df.x.max() - df.x.min()
    prev_gap = j.this_min - j.prev_max
    prev_gap_right = j.this_max - j.prev_min
    next_gap = j.next_min - j.this_max
    next_gap_left = j.next_max - j.this_min
    close_left = prev_gap < .30 and np.ceil(j.prev_min) == np.ceil(j.this_min)
    close_right = next_gap < .30 and np.ceil(j.next_max) == np.ceil(j.this_max)
    far_right = next_gap > .30 or np.abs(j.y_next) > 1.5 or next_gap_left < -1
    far_left = prev_gap > .30 or np.abs(j.y_prev) > 1.5 or prev_gap_right < -1
    # if the stroke is almost entirely in one box
    if (z == -1).all():
      Z = pd.Series(dict(x=mid_x-1, color="lightgray"))
    elif (z == -1).mean() > .75 and width < 1.5:
      Z = pd.Series(dict(x=mid_x-1, color="lightgray"))
    elif (z == -2).mean() > .75 and width < 1.5:
      Z = pd.Series(dict(x=mid_x-2, color="lightgray"))
    elif (z == 0).mean() > .75 and width < 1.5:
      Z = pd.Series(dict(x=mid_x, color="lightgray"))
    # if the stroke is closer to one side than .25, and further from the other
    # the number .25 is read off a histogram of gaps for unambiguous strokes
    elif close_left and not close_right:
      Z = pd.Series(dict(x=np.floor(j.this_min), color="red" if far_right else "orange"))
    elif close_right and not close_left:
      Z = pd.Series(dict(x=np.floor(j.this_max), color="green" if far_left else "blue"))
    else:
      Z = pd.Series(dict(color="black"))
      Z = z.value_counts(normalize=True)
      Z["color"] = "black"
      Z["x"] = mid_x - 1
    #for k, v in j.items():
    #  Z[k] = v
    Z["width"] = width
    Z["prev_gap"] = prev_gap
    Z["prev_gap_right"] = prev_gap_right
    Z["next_gap"] = next_gap
    Z["next_gap_left"] = next_gap_left
    Z["close_left"] = close_left
    Z["close_right"] = close_right
    Z["far_left"] = far_left
    Z["far_right"] = far_right
    return Z.rename(df.name).to_frame().T
  return get_ratio

def get_boxes(df, title):
  DX = 15
  df = rescale_and_cut(df)
  jumps = get_jumps(df)

  q = df.groupby("stroke_id").apply(ratio_factory(jumps))
  #if not (q.color == "black").any():
  if (q.color == "lightgray").all():
    return
  print(title)
  print(q[q.color != "lightgray"])
  jumps["x_small"] = jumps["this_min"] - jumps["this_max"] < 1
  jumps["y_med"] = df.groupby("stroke_id").y.median()
  jumps = jumps[np.floor(jumps["this_min"]) != np.floor(jumps["this_max"])]
  # so maybe only plot these?
  v2 = (1-df.symbol_digit)
  plt.figure(figsize=(12, 6))
  plt.plot(df.x + DX * v2, df.y, color="lightgray", zorder=-20)
  for idx, g in df.groupby("stroke_id"):
    qq = q.loc[idx]
    color = qq.color.values[0]
    v2 = (1-g.symbol_digit)
    plt.scatter(g.x + DX * v2, g.y, color=color, zorder=-10, alpha=0.2)
    plt.scatter(qq.x + 0.5 + DX * v2.max(), g.y.median(), color=color, zorder=-20, alpha=0.1, s=1000)
  plt.yticks(np.arange(-8, 2, step=2))
  plt.xticks(np.arange(0, 30))
  plt.gca().grid()
  plt.title(title)
  plt.show()