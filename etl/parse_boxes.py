import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

def rescale_and_cut(df):
  # these are magic numbers, deduced by trial and error
  # it absolutely can be done with scipy.optimize.fmin, but getting the objective right is hard
  df.x = df.x / 12.6 - 1.6
  df.y = df.y / 12.6 - 2.9 - (1 - df.symbol_digit) * 10.6
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
def aggregate_point_assignments(df):
  first_touch_box = np.floor(df.x.iloc[0]).astype(int)
  relative_box = np.floor(df.x) - first_touch_box
  literally_in_the_box = (relative_box == 0).mean()
  one_stroke = (np.abs(df.x - first_touch_box) < .07).mean()
  liminal = np.abs(df.x - np.round(df.x)) < 0.07
  relative_box = relative_box[~liminal]
  center = (relative_box == 0).mean()
  left = (relative_box == -1).mean()
  right = (relative_box == 1).mean()
  y_assigned = np.floor(df.y.median() / 2).astype(int)
  return pd.Series(dict(
    y_assigned=y_assigned,
    first_touch_box=first_touch_box, # for dumb historical reasons
    fraction_normal=literally_in_the_box,
    fraction_center=center,
    fraction_left=left,
    fraction_right=right,
    fraction_one=one_stroke))

def get_stroke_properties(df):
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

  jumps["prev_gap"] = jumps.this_min - jumps.prev_max
  jumps["prev_gap_right"] = jumps.this_max - jumps.prev_min
  jumps["next_gap"] = jumps.next_min - jumps.this_max
  jumps["next_gap_left"] = jumps.next_max - jumps.this_min

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

  jumps["width"] = jumps["this_max"] - jumps["this_min"]

  # if the stroke is closer to one side than .25, and further from the other
  # the number .25 is read off a histogram of gaps for unambiguous strokes
  jumps["close_left"] = (jumps.prev_gap < .30) & (np.ceil(jumps.prev_min) == np.ceil(jumps.this_min))
  jumps["close_right"] = (jumps.next_gap < .30) & (np.ceil(jumps.next_max) == np.ceil(jumps.this_max))
  jumps["far_right"] = (jumps.next_gap > .30) | (np.abs(jumps.y_next) > 1.5) | (jumps.next_gap_left < -1)
  jumps["far_left"] = (jumps.prev_gap > .30) | (np.abs(jumps.y_prev) > 1.5) | (jumps.prev_gap_right < -1)

  jumps = jumps.join(df.groupby("stroke_id").apply(aggregate_point_assignments))
  return jumps

# this should just return an assignment x, but the slow way is nice for debugging
def assign_stroke(j):
  # if the stroke is almost entirely in one literal box
  if j.fraction_normal > .75 and j.width < 1.5:
    Z = pd.Series(dict(x=j.first_touch_box, color="lightgray"))
  # if the stroke is almost entirely in one box, ignoring points near the edge
  elif j.fraction_center > .75 and j.width < 1.5:
    Z = pd.Series(dict(x=j.first_touch_box, color="lightgray"))
  elif j.fraction_left > .75 and j.width < 1.5:
    Z = pd.Series(dict(x=j.first_touch_box - 1, color="lightgray"))
  elif j.fraction_right > .75 and j.width < 1.5:
    Z = pd.Series(dict(x=j.first_touch_box + 1, color="lightgray"))
  elif j.close_left and not j.close_right:
    Z = pd.Series(dict(x=np.floor(j.this_min), color="lightgray" if j.far_right else "red"))
  elif j.close_right and not j.close_left:
    Z = pd.Series(dict(x=np.floor(j.this_max), color="lightgray" if j.far_left else "green"))
  # if the stroke is almost entirely along one edge
  elif j.fraction_one > .75 and j.width < 1.5:
    Z = pd.Series(dict(x=j.first_touch_box, color="blue"))
  else:
    Z = pd.Series(dict(x=j.first_touch_box, color="black"))
  for k in """y_assigned prev_gap prev_gap_right next_gap next_gap_left close_left close_right far_left far_right first_touch_box fraction_normal fraction_one fraction_left fraction_right fraction_center""".split():
    Z[k] = j[k]
  return Z

def debug_plot(points, assignments, title):
  DX = 15
  print(title)
  print(assignments[assignments.color != "lightgray"])
  plt.figure(figsize=(12, 6))
  z = (1 - points.symbol_digit)
  plt.plot(points.x + DX * z, points.y, color="lightgray", zorder=-20)
  for stroke_idx, stroke_points in points.groupby("stroke_id"):
    assignment = assignments.loc[stroke_idx]
    color = assignment.color
    z = (1 - stroke_points.symbol_digit)
    plt.scatter(stroke_points.x + DX * z, stroke_points.y, color=color, zorder=-10, alpha=0.2)
    plt.scatter(assignment.x + 0.5 + DX * z.max(), 2 * assignment.y_assigned + 0.5, color=color, zorder=-20, alpha=0.1, s=1000)
  plt.yticks(np.arange(-8, 2, step=2))
  plt.xticks(np.arange(0, 30))
  plt.gca().grid()
  plt.title(title)
  plt.show()

def get_boxes(df, title):
  df = rescale_and_cut(df)

  strokes = get_stroke_properties(df)
  assignments = strokes.apply(assign_stroke, axis=1)
  assignments = assignments.astype(int)
  print(title, (assignments.color != "lightgray").sum())
  #if (assignments.color != "lightgray").all():
  #  debug_plot(df, assignments, title)
  assignments = assignments["x y_assigned".split()]
  df = df.join(assignments, on="stroke_id", rsuffix="_assigned")
  return df