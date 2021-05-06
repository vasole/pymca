# /*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "Wout De Nolf"
__contact__ = "wout.de_nolf@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy
from PyMca5.PyMcaMath.fitting import Gefit
from PyMca5.PyMcaMath.fitting.Model import Model


class PolynomialModel(Model):
    def __init__(self, degree=0, maxiter=100):
        self._xdata = None
        self._ydata = None
        self._mask = None
        self._linear = True
        self.degree = degree
        self.maxiter = maxiter
        super(PolynomialModel, self).__init__()

    @property
    def degree(self):
        return self.coefficients.size - 1

    @degree.setter
    def degree(self, n):
        self.coefficients = numpy.zeros(n + 1)

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, values):
        self._coefficients = numpy.atleast_1d(values)

    @property
    def xdata(self):
        return self._xdata

    @xdata.setter
    def xdata(self, values):
        self._xdata = values

    @property
    def ydata(self):
        return self._ydata

    @ydata.setter
    def ydata(self, values):
        self._ydata = values

    @property
    def ystd(self):
        return None

    @property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, value):
        self._linear = value

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        self._maxiter = value


class LinearPolynomialModel(PolynomialModel):
    """y = c0 + c1*x + c2*x^2 + ..."""

    @property
    def _parameter_group_names(self):
        return ["coefficients"]

    @property
    def _linear_parameter_group_names(self):
        return ["coefficients"]

    def _iter_parameter_groups(self, linear_only=False):
        """
        :param bool linear_only:
        :yields (str, int): group name, nb. parameters in the group
        """
        if linear_only:
            names = self.linear_parameter_group_names
        else:
            names = self.parameter_group_names
        for name in names:
            if name == "coefficients":
                yield name, self.degree + 1

    def evaluate_fitmodel(self, xdata=None):
        """Evaluate the fit model, not the full model.

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        if xdata is None:
            xdata = self.xdata
        coeff = self.fit_parameters
        y = coeff[0] * numpy.ones_like(xdata)
        for i in range(1, len(coeff)):
            y += coeff[i] * (xdata ** i)
        return y

    def linear_derivatives_fitmodel(self, xdata=None):
        """Derivates to all linear parameters

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
        """
        if xdata is None:
            xdata = self.xdata
        return numpy.array(
            [self.derivative_fitmodel(i, xdata=xdata) for i in range(self.degree + 1)]
        )

    def derivative_fitmodel(self, param_idx, xdata=None):
        """Derivate to a specific parameter

        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        if xdata is None:
            xdata = self.xdata
        if param_idx == 0:
            return numpy.ones_like(xdata)
        else:
            return xdata ** param_idx


class ExponentialPolynomialModel(LinearPolynomialModel):
    """y = c0 * exp[c1*x + c2*x^2 + ...]
    yfit = log(y) = log(c1) + c1*x + c2*x^2 + ...
    """

    def _ydata_to_fit(self, ydata, xdata=None):
        return numpy.log(ydata)

    def _fit_to_ydata(self, yfit, xdata=None):
        return numpy.exp(yfit)

    def _parameters_to_fit(self, parameters):
        parameters = parameters.copy()
        parameters[0] = numpy.log(parameters[0])
        return parameters

    def _fit_to_parameters(self, parameters):
        parameters = parameters.copy()
        parameters[0] = numpy.exp(parameters[0])
        return parameters
