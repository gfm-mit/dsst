import argparse
import numpy as np
import pandas as pd
import tomli
import deepdiff


def get_setups(config):
  vv = parse_config(config)
  dd = {}
  for k, v in vv.iterrows():
    while k in dd:
      k += "#"
    dd[k] = v
  return dd

def pprint_dict(d, omit=None):
  str_dict = {}
  for k, v in d.items():
      if omit is not None and k in omit:
        continue
      if isinstance(v, str):
        str_dict[k] = v
      elif isinstance(v, int):
        str_dict[k] = str(v)
      elif 0 < v < .1:
        str_dict[k] = f"{v:.1e}"
      elif 0.9 < v < 1:
        str_dict[k] = f"1 - {1- v:.0e}"
      elif 0.1 <= v <= .9 or v == 0:
        str_dict[k] = f"{v:.2f}"
      else:
        str_dict[k] = f"{v:.2e}"
  return str(str_dict)

class TomlAction(argparse.Action):
    def __init__(self, toml_key=None, **kwargs):
        self.key = toml_key
        assert "default" not in kwargs
        kwargs["default"] = {}
        super(TomlAction, self).__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
      assert isinstance(values, str)
      with open(values, 'rb') as stream:
        config = tomli.load(stream)
      if self.key is not None:
        config = config[self.key]
      setattr(namespace, self.dest, config)

def parse_single_column(k, v):
  if isinstance(v, list):
    return pd.Series(v, name=k)
  assert isinstance(v, dict)
  if v.keys() == set("mid steps".split()):
    v["range"] = 10
  if v.keys() == set("mid steps range".split()):
    v = dict(low=v["mid"] / np.sqrt(v["range"]), high=v["mid"] * np.sqrt(v["range"]), steps=v["steps"])
  assert "low" in v.keys()
  assert "high" in v.keys()
  assert "steps" in v.keys()
  assert not v.keys() - set("low high steps int".split())
  generated = np.geomspace(v["low"], v["high"], v["steps"])
  if "int" in v.keys():
    generated = generated.astype(int)
  return pd.Series(generated.tolist(), name=k)

def cross_reduce(*dfs):
  accum, *rest = dfs
  accum = accum.to_frame()
  for column in rest:
    accum = accum.merge(column, how='cross')
  return accum

def parse_column_group(group, cartesian=False):
  assert isinstance(group, dict)
  column_group = [parse_single_column(k, v) for k, v in group.items()]
  if not column_group:
    return None

  if cartesian:
    column_group = cross_reduce(*column_group)
  else:
    column_lengths = {kv.name: kv.shape[0] for kv in column_group}
    any_length = list(column_lengths.values())[0]
    assert all([x == any_length for x in column_lengths.values()]), column_lengths

    column_group = pd.DataFrame(column_group).transpose()
  return column_group

def parse_config(config):
  param_sets = []
  assert not config.keys() - "row_major column_major scalar meta".split()
  row_major = config.get("row_major", [])
  column_major = config.get("column_major", {})
  scalar = config.get("scalar", {})
  meta = config.get("meta", {})
  assert isinstance(meta, dict)
  assert not meta.keys() - "repeat shuffle cartesian overlay append".split()
  meta_repeat = meta.get("repeat", None)
  meta_shuffle = meta.get("shuffle", None)
  meta_cartesian = meta.get("cartesian", None)
  meta_overlay = meta.get("overlay", None)
  meta_append = meta.get("append", None)
  assert not (meta_overlay and meta_append)

  row_count = None

  column_major = parse_column_group(column_major, cartesian=meta_cartesian)
  if column_major is not None:
    param_sets += [column_major]
    row_count = column_major.shape[0]

  if meta_append:
    assert row_major and row_count > 0
    param_sets = [pd.concat([column_major, pd.DataFrame(row_major)], axis=0)]
    if scalar:
      scalar = pd.DataFrame([scalar] * row_count)
      param_sets += [scalar]
  elif row_major:
    if row_count is not None:
      assert len(row_major) == row_count, f"{len(row_major)=} != {column_major.shape=}"
    else:
      if not row_major:
        row_major = [{}]
      row_count = len(row_major)

    if meta_overlay:
      row_major = pd.DataFrame([scalar | row for row in row_major])
    else:
      row_major = pd.DataFrame([dict(**scalar, **row) for row in row_major])
    param_sets += [row_major]
  elif scalar is not None:
    if row_count is None:
      row_count = 1
    scalar = pd.DataFrame([scalar] * row_count)
    param_sets += [scalar]
  param_sets = pd.concat(param_sets, axis=1)

  duplicate_columns = param_sets.columns[param_sets.columns.duplicated()].values
  assert not duplicate_columns.shape[0], duplicate_columns
  constant_columns = [
    k for k in param_sets.columns
    if param_sets[k].nunique() == 1 # inefficient, but not load bearing
  ]

  assert meta_repeat is None or isinstance(meta_repeat, int)
  assert meta_shuffle is None or isinstance(meta_shuffle, bool)
  if meta_repeat:
    param_sets = pd.concat([param_sets] * meta_repeat, axis=0, ignore_index=True)
  if meta_shuffle:
    assert not meta_cartesian
    for k in param_sets.columns:
      param_sets.loc[:, k] = np.random.permutation(param_sets.loc[:, k].values)

  if row_count == 1:
    param_sets.index = ["{}"] * param_sets.shape[0]
  else:
    param_sets.index = [
      pprint_dict(row.to_dict(), omit=constant_columns)
      for _, row in param_sets.iterrows()
    ]
  cum_idx = (param_sets.groupby(param_sets.index).cumcount() + 1).astype(str)
  cum_idx = cum_idx.apply(lambda x: "" if x == "1" else "#" + x)
  param_sets.index = param_sets.index + cum_idx
  return param_sets

def test(f_in, f_out):
  with open(f_in, 'rb') as stream:
    config = tomli.load(stream)
    parsed = parse_config(config)
    parsed = [row.to_dict() for _, row in parsed.iterrows()]
    for p in parsed:
      print(p)
  with open(f_out, 'rb') as stream:
    expected = tomli.load(stream)['params']
  print(deepdiff.DeepDiff(expected, parsed, ignore_order=False).pretty() or "NO DIFF")

# no, bad, don't do testing this way!
if __name__ == "__main__":
  test('./util/test/base_in.toml', './util/test/base_out.toml')
  test('./util/test/overlay_in.toml', './util/test/overlay_out.toml')
  test('./util/test/append_in.toml', './util/test/append_out.toml')