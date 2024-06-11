from abc import ABC, abstractmethod


class Base(ABC):
  def __init__(self, device='cpu'):
    self.device = device

  @abstractmethod
  def get_classifier_architecture(self):
    pass

  @abstractmethod
  def get_parameters(self, **kwargs):
    pass

  @abstractmethod
  def get_tuning_ranges(self):
    pass

  @abstractmethod
  def get_coefficients(self, model):
    pass


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