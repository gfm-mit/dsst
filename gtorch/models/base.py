from abc import ABC, abstractmethod

class Base(ABC):
  def __init__(self, device='cpu'):
    self.device = device

  @abstractmethod
  def get_architecture(self):
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