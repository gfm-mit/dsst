import numpy as np
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.transforms import Transform
from scipy.stats import norm
from matplotlib.ticker import NullFormatter, LogitLocator, LogitFormatter, AutoLocator, FuncFormatter

class LogOneMinusXScale(ScaleBase):
    """
    Scale for the log of one minus x: log(1 - x).
    """

    name = 'log1minusx'

    class LogOneMinusXTransform(Transform):
        input_dims = output_dims = 1

        def transform_non_affine(self, values):
            # Applying the transformation log(1 - x)
            with np.errstate(divide='ignore', invalid='ignore'):
                return -np.log(1 - values)

        def inverted(self):
            return LogOneMinusXScale.InvertedLogOneMinusXTransform()

    class InvertedLogOneMinusXTransform(Transform):
        input_dims = output_dims = 1

        def transform_non_affine(self, values):
            # Inverse transformation: exp(values) + x - 1
            return 1 - np.exp(-values)

        def inverted(self):
            return LogOneMinusXScale.LogOneMinusXTransform()

    def get_transform(self):
        return self.LogOneMinusXTransform()

    def set_default_locators_and_formatters(self, axis):
        # The locator finds this scale's "nice" tick locations, AutoLocator will be usually good
        axis.set_major_locator(LogitLocator())
        axis.set_minor_locator(LogitLocator())

        # The formatter takes the tick locations and formats them to the desired strings
        axis.set_major_formatter(LogitFormatter())
        axis.set_minor_formatter(NullFormatter())  # No labels on the minor ticks