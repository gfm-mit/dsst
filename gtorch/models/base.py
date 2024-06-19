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


class SequenceBase(Base):
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