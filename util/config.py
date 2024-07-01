import argparse
import numpy as np
import pandas as pd
import tomli
import deepdiff
import pprint


def get_spaces(**kwargs):
  print("spaces.kwargs:")
  for k, v in kwargs.items():
    print(f"  {k}: {v}")
  spaces = pd.DataFrame(kwargs)
  for col in spaces.columns:
    spaces[col] = np.random.permutation(spaces[col].values)
  return spaces


def postprocess_tuning_ranges(tuning_ranges):
  overlay = {}
  multiple = 1
  if "multiple" in tuning_ranges:
    multiple = tuning_ranges.pop("multiple")
  for k in tuning_ranges.keys():
    if isinstance(tuning_ranges[k], dict):
      if tuning_ranges[k].keys() == set("low high steps".split()):
        tuning_ranges[k] = np.geomspace(
          tuning_ranges[k]["low"],
          tuning_ranges[k]["high"],
          tuning_ranges[k]["steps"]
        ).tolist()
      elif tuning_ranges[k].keys() == set("low high steps int".split()):
        tuning_ranges[k] = np.geomspace(
          tuning_ranges[k]["low"],
          tuning_ranges[k]["high"],
          tuning_ranges[k]["steps"]
        ).astype(int).tolist()
      elif tuning_ranges[k].keys() == set("multiple values".split()):
        tuning_ranges[k] = tuning_ranges[k]["values"] * tuning_ranges[k]["multiple"]
      else:
        assert False, tuning_ranges[k]
    elif isinstance(tuning_ranges[k], list):
      pass
    else:
      overlay[k] = tuning_ranges.pop(k)
  if multiple != 1:
    for k, v in tuning_ranges.items():
      tuning_ranges[k] = tuning_ranges[k] * multiple
  return overlay, tuning_ranges


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

def parse_column(kv):
  k, v = kv
  if isinstance(v, list):
    return pd.Series(v, name=k)
  assert isinstance(v, dict)
  assert "low" in v.keys()
  assert "high" in v.keys()
  assert "steps" in v.keys()
  assert not v.keys() - set("low high steps int".split())
  generated = np.geomspace(v["low"], v["high"], v["steps"])
  if "int" in v.keys():
    generated = generated.astype(int)
  return pd.Series(generated.tolist(), name=k)

def parse_config(config):
  param_sets = []
  row_major = config.get("row_major", None)
  column_major = config.get("column_major", None)
  scalar = config.get("scalar", None)
  meta = config.get("meta", None)
  if row_major is not None:
    row_major = pd.DataFrame(row_major)
    param_sets += [row_major]
  if column_major is not None:
    column_major = map(parse_column, column_major.items())
    column_major = pd.DataFrame(column_major).transpose()
    param_sets += [column_major]
  assert row_major is None or column_major is None or row_major.shape[0] == column_major.shape[0]
  row_count = row_major.shape[0] if row_major is not None else column_major.shape[0] if column_major is not None else 1
  if scalar is not None:
    scalar = pd.DataFrame([scalar] * row_count)
    param_sets += [scalar]
  elif row_major is None and column_major is None:
    scalar = pd.DataFrame([{}])
    param_sets += [scalar]
  param_sets = pd.concat(param_sets, axis=1)
  if (meta | {}).get("repeat", None):
    param_sets = pd.concat([param_sets] * meta["repeat"], axis=0, ignore_index=True)
  return [row.to_dict() for _, row in param_sets.iterrows()]

# no, bad, don't do testing this way!
if __name__ == "__main__":
  with open('./util/test_in.toml', 'rb') as stream:
    config = tomli.load(stream)
    parsed = parse_config(config)
  with open('./util/test_out.toml', 'rb') as stream:
    expected = tomli.load(stream)['params']
  pprint.pprint(deepdiff.DeepDiff(expected, parsed, ignore_order=False))