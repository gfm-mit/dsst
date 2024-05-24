import numpy as np
from matplotlib.scale import ScaleBase
from matplotlib.ticker import LogitFormatter, LogitLocator, NullFormatter
from matplotlib.transforms import Transform
from scipy.stats import norm


class ProbitTransform(Transform):
    input_dims = output_dims = 1

    def transform_non_affine(self, values):
        # Applying the inverse CDF (probit function)
        with np.errstate(divide='ignore', invalid='ignore'):
            return norm.ppf(values)

    def inverted(self):
        return ProbitInverseTransform()

class ProbitInverseTransform(Transform):
    input_dims = output_dims = 1

    def transform_non_affine(self, values):
        # Applying the CDF
        return norm.cdf(values)

    def inverted(self):
        return ProbitTransform()

class ProbitScale(ScaleBase):
    name = 'probit'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)

    def get_transform(self):
        return ProbitTransform()

    def set_default_locators_and_formatters(self, axis):
        # The locator finds this scale's "nice" tick locations, AutoLocator will be usually good
        axis.set_major_locator(LogitLocator())
        axis.set_minor_locator(LogitLocator())

        # The formatter takes the tick locations and formats them to the desired strings
        axis.set_major_formatter(LogitFormatter())
        axis.set_minor_formatter(NullFormatter())  # No labels on the minor ticks