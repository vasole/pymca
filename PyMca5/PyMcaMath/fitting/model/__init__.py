"""Base classes for fit models and combined fit models
"""

from PyMca5.PyMcaMath.fitting.model.ParameterModel import (
    nonlinear_parameter_group,
    independent_linear_parameter_group,
)
from PyMca5.PyMcaMath.fitting.model.LeastSquaresFitModel import (
    LeastSquaresFitModel,
    LeastSquaresCombinedFitModel,
)
