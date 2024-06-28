from abc import ABC, abstractmethod


class Base(ABC):
  def __init__(self, device=None):
    assert device is not None
    self.device = device

  @abstractmethod
  def get_classifier_architecture(self):
    pass

  @abstractmethod
  def get_parameters(self, **kwargs):
    pass

  def get_coefficients(self, model):
    assert "get_coefficients not implemented in {}".format(self.__class__.__name__)

  def translate_state_dict(self, state_dict):
    print("falling back on default state_dict translation: cutting off last tunable layer")
    z = max([int(x.split(".")[0]) for x in state_dict.keys()])
    z = f"{z}."
    for k in list(state_dict.keys()):
      if k.startswith(z):
        del state_dict[k]
    return state_dict

class SequenceBase(Base):
  @abstractmethod
  def translate_state_dict(self, state_dict):
    pass

  @abstractmethod
  def get_next_token_architecture(self):
    pass

  @abstractmethod
  def get_classifier_parameters(self):
    pass

  @abstractmethod
  def get_next_token_parameters(self):
    pass

  def get_parameters(self, **kwargs):
    if "task" in kwargs and kwargs["task"] == "next_token":
      return self.get_classifier_parameters() | self.get_next_token_parameters()
    return self.get_classifier_parameters()