import argparse
import numpy as np
import pandas as pd
import tomli


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