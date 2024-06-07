from abc import ABC, abstractmethod


class Base(ABC):
  def __init__(self, device='cpu'):
    self.device = device

  @abstractmethod
  def get_classifier_architecture(self):
    pass

  @abstractmethod
  def get_parameters(self):
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