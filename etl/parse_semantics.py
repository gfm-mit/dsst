import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

def load_semantics():
  script_dir = os.path.dirname(__file__) # absolute dir the script is in
  rel_path = "schema/box_semantics.csv"
  abs_file_path = os.path.join(script_dir, rel_path)
  df = pd.read_csv(abs_file_path, header=None).fillna(-1).astype(int)
  df.index = [3,2,1,0] # backwards of the row numbers in the file
  #print("semantics\n", df, "\nsemantics")
  df = df.transpose().unstack()
  return df

def assign_semantics(df, semantics):
  df["debug_x"] = df.x + 15 * (1 - df.symbol_digit)
  df["debug_y"] = df.y
  df.x -= df.x_assigned
  df.y -= df.y_assigned

  df.y_assigned = df.y_assigned.astype(int) + 4
  df.x_assigned = df.x_assigned.astype(int)
  is_last_six = (1 * df.symbol_digit) * (df.y_assigned == 0) * (df.x_assigned >= 6)
  df.x_assigned += is_last_six * 2
  df["symbol_assigned"] = semantics.reindex(
    pd.MultiIndex.from_frame(df["y_assigned x_assigned".split()]),
    fill_value=-1).values
  df["box_assigned"] = (3 - df.y_assigned) * 14 + df.x_assigned
  df.loc[df.symbol_assigned.isna(), "box_assigned"] = np.nan
  df.loc[df.symbol_assigned.isna(), "box_assigned"] = np.nan
  df.loc[df.symbol_assigned.isna(), "box_assigned"] -= 2
  df = df.dropna(subset=['box_assigned'])
  df.box_assigned = df.box_assigned.astype(int)

  df["debug_t"] = df.t
  previous_row = pd.Series(df.t.values, index=np.roll(df.stroke_id, -1))
  previous_row = previous_row[previous_row.index != df.stroke_id.values]
  previous_row.iloc[-1] = 0
  df.t = df.t - previous_row.loc[df.stroke_id].values
  df["task_assigned"] = (df.box_assigned >= 6) * 1 + (1-df.symbol_digit) * 2 + (df.box_assigned >= 48) * (1-df.symbol_digit)
  df = df["x y t task_assigned box_assigned symbol_assigned".split()]
  return df