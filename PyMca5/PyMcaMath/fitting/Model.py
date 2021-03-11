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


import functools
from collections.abc import Sequence, MutableMapping
import numpy
from contextlib import contextmanager, ExitStack
from PyMca5.PyMcaMath.linalg import lstsq
from PyMca5.PyMcaMath.fitting import Gefit


def enable_caching(method):
    @functools.wraps(method)
    def cache_wrapper(self, *args, **kw):
        with self.caching_context():
            return method(self, *args, **kw)

    return cache_wrapper


def memoize(method):
    @functools.wraps(method)
    def cache_wrapper(self):
        if self.caching_enabled:
            name = method.__qualname__
            if name in self._cache:
                return self._cache[name]
            else:
                r = self._cache[name] = method(self)
                return r

    return cache_wrapper


def memoize_property(method):
    return property(memoize(method))


class Cashed(object):
    def __init__(self):
        self._cache = None

    @contextmanager
    def caching_context(self):
        reset = self._cache is None
        if reset:
            self._cache = {}
        try:
            yield
        finally:
            if reset:
                self._cache = None

    @property
    def caching_enabled(self):
        return self._cache is not None


class Model(Cashed):
    """Evaluation and derivatives of a model to be used in least-squares fitting."""

    def __init__(self):
        self.included_parameters = None
        self.excluded_parameters = None
        super(Model, self).__init__()

    @property
    def xdata(self):
        raise AttributeError from NotImplementedError

    @property
    def ydata(self):
        raise AttributeError from NotImplementedError

    @property
    def ystd(self):
        raise AttributeError from NotImplementedError

    @property
    def yfitdata(self):
        return self._ydata_to_fit(self.ydata)

    @property
    def yfitstd(self):
        return self._ystd_to_fit(self.ystd)

    @property
    def yfullmodel(self):
        """Model of ydata"""
        return self.evaluate_fullmodel()

    @property
    def yfitmodel(self):
        """Model of yfitdata"""
        return self.evaluate_fitmodel()

    @property
    def nchannels(self):
        raise AttributeError from NotImplementedError

    @property
    def fit_parameters(self):
        return self._get_parameters(fitting=True)

    @fit_parameters.setter
    def fit_parameters(self, values):
        return self._set_parameters(values, fitting=True)

    @property
    def parameters(self):
        return self._get_parameters(fitting=False)

    @parameters.setter
    def parameters(self, values):
        return self._set_parameters(values, fitting=False)

    @property
    def constraints(self):
        return self._get_constraints()

    @property
    def nparameters(self):
        return sum(tpl[1] for tpl in self._parameter_groups())

    @property
    def parameter_names(self):
        return list(self._iter_parameter_names())

    @property
    def parameter_group_names(self):
        return self._filter_parameter_names(self._parameter_group_names)

    @property
    def _parameter_group_names(self):
        raise AttributeError from NotImplementedError

    @property
    def linear_fit_parameters(self):
        return self._get_parameters(linear_only=True, fitting=True)

    @linear_fit_parameters.setter
    def linear_fit_parameters(self, params):
        return self._set_parameters(params, linear_only=True, fitting=True)

    @property
    def linear_parameters(self):
        return self._get_parameters(linear_only=True, fitting=False)

    @linear_parameters.setter
    def linear_parameters(self, params):
        return self._set_parameters(params, linear_only=True, fitting=False)

    @property
    def linear_constraints(self):
        return self._get_constraints(linear_only=True)

    @property
    def nlinear_parameters(self):
        return sum(tpl[1] for tpl in self._parameter_groups(linear_only=True))

    @property
    def linear_parameter_names(self):
        return list(self._iter_parameter_names(linear_only=True))

    @property
    def linear_parameter_group_names(self):
        return self._filter_parameter_names(self._linear_parameter_group_names)

    @property
    def _linear_parameter_group_names(self):
        raise AttributeError from NotImplementedError

    def _get_parameters(self, linear_only=False, fitting=True):
        """
        :param bool linear_only:
        :param bool fitting:
        :returns array:
        """
        i = 0
        if linear_only:
            nparams = self.nlinear_parameters
        else:
            nparams = self.nparameters
        params = numpy.zeros(nparams)
        for name, n in self._parameter_groups(linear_only=linear_only):
            params[i : i + n] = getattr(self, name)
            i += n
        if fitting:
            return self._parameters_to_fit(params)
        else:
            return params

    def _get_constraints(self, linear_only=False):
        """
        :param bool linear_only:
        :returns array:
        """
        i = 0
        if linear_only:
            nparams = self.nlinear_parameters
        else:
            nparams = self.nparameters
        codes = numpy.zeros((nparams, 3), numpy.float64)
        bspecified = False
        for name, n in self._parameter_groups(linear_only=linear_only):
            name += "_constraint"
            if hasattr(self, name):
                bspecified = True
                codes[i : i + n] = getattr(self, name)
            i += n
        if bspecified:
            return codes.T
        else:
            return None

    def _set_parameters(self, params, linear_only=False, fitting=False):
        """
        :param bool linear_only:
        :param bool fitting:
        """
        if fitting:
            params = self._fit_to_parameters(params)
        i = 0
        for name, n in self._parameter_groups(linear_only=linear_only):
            if n > 1:
                getattr(self, name)[:] = params[i : i + n]
            elif n == 1:
                setattr(self, name, params[i])
            i += n

    def _filter_parameter_names(self, names):
        included = self.included_parameters
        excluded = self.excluded_parameters
        if included is None:
            included = names
        if excluded is None:
            excluded = []
        return [name for name in names if name in included and name not in excluded]

    def evaluate_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        return self._fit_to_ydata(self.evaluate_fitmodel(xdata=xdata))

    def evaluate_linear_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        return self._fit_to_ydata(self.evaluate_linear_fitmodel(xdata=xdata))

    def evaluate_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        raise NotImplementedError

    def evaluate_linear_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        derivatives = self.linear_derivatives_fitmodel(xdata=xdata)
        return self.linear_parameters.dot(derivatives)

    def linear_decomposition_fitmodel(self, xdata=None):
        """Linear decomposition of the fit model.

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
        """
        derivatives = self.linear_derivatives_fitmodel(xdata=xdata)
        return self.linear_parameters[:, numpy.newaxis] * derivatives

    def derivative_fitmodel(self, param_idx, xdata=None):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        raise NotImplementedError

    def derivatives_fitmodel(self, xdata=None):
        """Derivates to all parameters of the fit model.

        :param array xdata: length nxdata
        :returns list(array): nparams x nxdata
        """
        if xdata is None:
            xdata = self.xdata
        return [
            self.derivative_fitmodel(i, xdata=xdata) for i in range(self.nparameters)
        ]

    def linear_derivatives_fitmodel(self, xdata=None):
        """Derivates to all linear parameters

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
        """
        raise NotImplementedError

    @property
    def linear(self):
        raise AttributeError from NotImplementedError

    def _iter_parameter_groups(self, linear_only=False):
        """
        :param bool linear_only:
        :yields (str, int): group name, nb. parameters in the group
        """
        raise NotImplementedError

    def _parameter_groups(self, linear_only=False):
        """
        :param bool linear_only:
        :returns iterable(str, int): group name, nb. parameters in the group
        """
        if self.caching_enabled:
            cache = self._cache.setdefault("all_parameter_groups", {})
            a = self.included_parameters
            b = self.excluded_parameters
            if a is not None:
                a = tuple(sorted(a))
            if b is not None:
                b = tuple(sorted(b))
            key = a, b
            it = cache.get(key)
            if it is None:
                it = cache[key] = list(
                    self._iter_parameter_groups(linear_only=linear_only)
                )
        else:
            it = self._iter_parameter_groups(linear_only=linear_only)
        return it

    def _iter_parameter_names(self, linear_only=False):
        """
        :param bool linear_only:
        :yields str:
        """
        for name, n in self._parameter_groups(linear_only=linear_only):
            if n > 1:
                for i in range(n):
                    yield name + str(i)
            else:
                yield name

    def _parameter_name_from_index(self, idx, linear_only=False):
        """Parameter index to group name and group index

        :returns str, int: group name, index in parameter group
        """
        i = 0
        for name, n in self._parameter_groups(linear_only=linear_only):
            if idx >= i and idx < (i + n):
                return name, idx - i
            i += n

    def fit(self, full_output=False):
        """
        :param bool full_output: add statistics to fitted parameters
        :returns dict:
        """
        if self.linear:
            return self.linear_fit(full_output=full_output)
        else:
            return self.nonlinear_fit(full_output=full_output)

    @enable_caching
    def linear_fit(self, full_output=False):
        """
        :param bool full_output: add statistics to fitted parameters
        :returns dict:
        """
        if self.niter_non_leastsquares:
            keep = self.linear_parameters
        try:
            b = self.yfitdata  # nchannels
            for i in range(max(self.niter_non_leastsquares, 1)):
                A = self.linear_derivatives_fitmodel().T  # nchannels, nparams
                result = lstsq(A, b.copy(), digested_output=full_output)
                if self.niter_non_leastsquares:
                    self.linear_fit_parameters = result[0]
                    self.non_leastsquares_increment()
        finally:
            if self.niter_non_leastsquares:
                self.linear_parameters = keep
        return {
            "linear": True,
            "parameters": self._fit_to_parameters(result[0]),
            "uncertainties": self._fit_to_uncertainties(result[1]),
        }

    @enable_caching
    def nonlinear_fit(self, full_output=False):
        """
        :param bool full_output: add statistics to fitted parameters
        :returns dict:
        """
        keep = self.parameters
        constraints = self.constraints
        xdata = self.xdata
        ydata = self.yfitdata
        ystd = self.yfitstd
        try:
            for i in range(max(self.niter_non_leastsquares, 1)):
                result = Gefit.LeastSquaresFit(
                    self._evaluate_fitmodel,
                    self.fit_parameters,
                    model_deriv=self._derivative_fitmodel,
                    xdata=xdata,
                    ydata=ydata,
                    sigmadata=ystd,
                    constrains=constraints,
                    maxiter=self.maxiter,
                    weightflag=self.weightflag,
                    deltachi=self.deltachi,
                    fulloutput=full_output,
                )
                if self.niter_non_leastsquares:
                    self.fit_parameters = parameters
                    self.non_leastsquares_increment()
        finally:
            self.parameters = keep
        ret = {
            "linear": False,
            "parameters": self._fit_to_parameters(result[0]),
            "uncertainties": self._fit_to_uncertainties(result[2]),
            "chi2_red": result[1],
        }
        if full_output:
            ret["niter"] = result[3]
            ret["lastdeltachi"] = result[4]
        return ret

    def _ydata_to_fit(self, ydata):
        return ydata

    def _ystd_to_fit(self, ystd):
        return ystd

    def _parameters_to_fit(self, params):
        return params

    def _fit_to_ydata(self, yfit):
        return yfit

    def _fit_to_parameters(self, params):
        return params

    def _fit_to_uncertainties(self, uncertainties):
        return uncertainties

    def _linear_parameters_to_fit(self, params):
        return params

    def _fit_to_linear_parameters(self, params):
        return params

    @property
    def maxiter(self):
        return 100

    @property
    def deltachi(self):
        return None

    @property
    def weightflag(self):
        return 0

    def _evaluate_fitmodel(self, parameters, xdata):
        """Update parameters and evaluate model

        :param array parameters: length nparams
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        self.fit_parameters = parameters
        return self.evaluate_fitmodel(xdata=xdata)

    def _derivative_fitmodel(self, parameters, param_idx, xdata):
        """Update parameters and return derivate to a specific parameter

        :param array parameters: length nparams
        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        self.fit_parameters = parameters
        return self.derivative_fitmodel(param_idx, xdata=xdata)

    def use_fit_result(self, result):
        """
        :param dict result:
        """
        if result["linear"]:
            self.linear_parameters = result["parameters"]
        else:
            self.parameters = result["parameters"]

    @property
    def niter_non_leastsquares(self):
        return 0

    def non_leastsquares_increment(self):
        raise NotImplementedError


class ConcatModel(Model):
    """Concatenated model with shared parameters"""

    def __init__(self, models, shared_attributes=None):
        if not isinstance(models, Sequence):
            models = [models]
        for model in models:
            if not isinstance(model, Model):
                raise ValueError("'models' must be a list of type 'Model'")
        self._models = models
        self.__fixed_shared_attributes = {
            "linear",
            "included_parameters",
            "excluded_parameters",
        }
        self.shared_attributes = shared_attributes
        super(ConcatModel, self).__init__()

    @property
    def model(self):
        """Model used to get/set shared attributes"""
        return self._models[0]

    def __getattr__(self, name):
        """Get shared attribute"""
        if self.nmodels and name in self.shared_attributes:
            return getattr(self.model, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        """Set shared attribute"""
        if (
            name != "_models"
            and self.nmodels
            and hasattr(self.model, name)
            and name in self.shared_attributes
        ):
            for m in self._models:
                setattr(m, name, value)
        else:
            super(ConcatModel, self).__setattr__(name, value)

    @property
    def shared_attributes(self):
        """Attributes shared between the fit models (they should have the same value)"""
        return self._shared_attributes

    @shared_attributes.setter
    def shared_attributes(self, shared_attributes):
        """
        :param Sequence(str) shared_attributes:
        """
        if shared_attributes is None:
            shared_attributes = set()
        else:
            shared_attributes = set(shared_attributes)
        shared_attributes |= self.__fixed_shared_attributes
        if self.nmodels <= 1:
            self._shared_attributes = shared_attributes
            return
        self.share_attributes(shared_attributes)
        self.validate_shared_attributes(shared_attributes)
        self._shared_attributes = shared_attributes

    def validate_shared_attributes(self, shared_attributes=None):
        """Check whether attributes are shared

        :param Sequence(str) shared_attributes:
        :raises AssertionError:
        """
        if self.nmodels <= 1:
            return
        if shared_attributes is None:
            shared_attributes = self._shared_attributes
        for name in shared_attributes:
            value = getattr(self.model, name)
            if isinstance(value, (Sequence, MutableMapping, numpy.ndarray)):
                for m in self._models[1:]:
                    assert id(value) == id(getattr(m, name)), name
            else:
                for m in self._models[1:]:
                    assert value == getattr(m, name), name

    def share_attributes(self, shared_attributes=None):
        """Ensure attributes are shared

        :param Sequence(str) shared_attributes:
        """
        if self.nmodels <= 1:
            return
        if shared_attributes is None:
            shared_attributes = self._shared_attributes
        model = self.model
        adict = {name: getattr(model, name) for name in shared_attributes}
        for model in self._models[1:]:
            for name, value in adict.items():
                setattr(model, name, value)

    @property
    def nmodels(self):
        return len(self._models)

    @property
    def nchannels(self):
        nmodels = self.nmodels
        if nmodels == 0:
            return 0
        else:
            return sum([m.nchannels for m in self._models])

    @property
    def xdata(self):
        return self._get_data("xdata")

    @xdata.setter
    def xdata(self, values):
        self._set_data("xdata", values)

    @property
    def ydata(self):
        return self._get_data("ydata")

    @ydata.setter
    def ydata(self, values):
        self._set_data("ydata", values)

    @property
    def ystd(self):
        return self._get_data("ystd")

    @ystd.setter
    def ystd(self, values):
        self._set_data("ystd", values)

    @property
    def yfitdata(self):
        return self._get_data("yfitdata")

    @property
    def yfitstd(self):
        return self._get_data("yfitstd")

    def _get_data(self, attr):
        """
        :param str attr:
        :returns array:
        """
        nmodels = self.nmodels
        if nmodels == 0:
            return None
        elif nmodels == 1:
            return getattr(self.model, attr)
        elif getattr(self.model, attr) is None:
            return None
        else:
            return numpy.concatenate([getattr(m, attr) for m in self._models])

    def _set_data(self, attr, values):
        """
        :param str attr:
        :param array values:
        """
        if len(values) != self.nchannels:
            raise ValueError("Not the expected number of channels")
        for idx, model in self._iter_models(values):
            setattr(model, attr, values[idx])

    @contextmanager
    def _filter_parameter_context(self, shared=True):
        keepex = self.excluded_parameters
        keepinc = self.included_parameters
        try:
            if shared:
                if keepinc:
                    self.included_parameters = list(
                        set(keepinc) - set(self.shared_attributes)
                    )
                else:
                    self.included_parameters = self.shared_attributes
            else:
                if keepex:
                    self.excluded_parameters.extend(self.shared_attributes)
                else:
                    self.excluded_parameters = self.shared_attributes
            yield
        finally:
            self.excluded_parameters = keepex
            self.included_parameters = keepinc

    @property
    def nparameters(self):
        return sum(m.nparameters for m in self._iter_parameter_models())

    @property
    def nlinear_parameters(self):
        return sum(m.nlinear_parameters for m in self._iter_parameter_models())

    @property
    def nshared_parameters(self):
        with self._filter_parameter_context(shared=True):
            return self.model.nparameters

    @property
    def nshared_linear_parameters(self):
        with self._filter_parameter_context(shared=True):
            return self.model.nlinear_parameters

    def _get_parameters(self, linear_only=False, fitting=False):
        """
        :param bool linear_only:
        :returns array:
        """
        return numpy.concatenate(
            [
                m._get_parameters(linear_only=linear_only, fitting=fitting)
                for m in self._iter_parameter_models()
            ]
        )

    def _set_parameters(self, values, linear_only=False, fitting=False):
        """
        :paramm array values:
        :param bool linear_only:
        """
        i = 0
        for m in self._iter_parameter_models():
            if linear_only:
                n = m.nlinear_parameters
            else:
                n = m.nparameters
            if n:
                m._set_parameters(
                    values[i : i + n], linear_only=linear_only, fitting=fitting
                )
                i += n

    def _iter_parameter_models(self):
        """Iterate over models which are temporarily configured so that
        after iterations, all parameters provided.
        :yields Model:
        """
        with self._filter_parameter_context(shared=True):
            yield self.model
        with self._filter_parameter_context(shared=False):
            for m in self._models:
                yield m

    def _parameter_groups(self, linear_only=False):
        """
        :param bool linear_only:
        :yields (str, int): group name, nb. parameters in the group
        """
        with self._filter_parameter_context(shared=True):
            for item in self.model._parameter_groups(linear_only=linear_only):
                yield item
        with self._filter_parameter_context(shared=False):
            for i, m in enumerate(self._models):
                for name, n in self.model._parameter_groups(linear_only=linear_only):
                    yield name + str(i), n

    def _parameter_model_index(self, idx, linear_only=False):
        """Convert parameter index of ConcatModel to a parameter indices
        of the underlying models (only one when parameter is not shared).

        :param bool linear_only:
        :param int idx:
        :returns iterable(tuple): model index, parameter index in this model
        """
        if self.caching_enabled:
            cache = self._cache.setdefault("parameter_model_index", {})
            it = cache.get(idx)
            if it is None:
                it = cache[idx] = list(
                    self._iter_parameter_index(idx, linear_only=linear_only)
                )
        else:
            it = self._iter_parameter_index(idx, linear_only=linear_only)
        return it

    def _iter_parameter_index(self, idx, linear_only=False):
        """Convert parameter index of ConcatModel to a parameter indices
        of the underlying models (only one when parameter is not shared).

        :param bool linear_only:
        :param int idx:
        :yields (int, int): model index, parameter index in this model
        """
        if linear_only:
            nshared = self.nshared_linear_parameters
        else:
            nshared = self.nshared_parameters
        shared_attributes = self.shared_attributes
        if idx < nshared:
            for i, m in enumerate(self._models):
                iglobal = 0
                imodel = 0
                for name, n in m._parameter_groups(linear_only=linear_only):
                    if name in shared_attributes:
                        if idx >= iglobal and idx < (iglobal + n):
                            yield i, imodel + idx - iglobal
                        iglobal += n
                    imodel += n
        else:
            iglobal = nshared
            for i, m in enumerate(self._models):
                imodel = 0
                for name, n in m._parameter_groups(linear_only=linear_only):
                    if name not in shared_attributes:
                        if idx >= iglobal and idx < (iglobal + n):
                            yield i, imodel + idx - iglobal
                            return
                        iglobal += n
                    imodel += n

    @property
    def shared_parameters(self):
        with self._filter_parameter_context(shared=True):
            return self.model.parameters

    @shared_parameters.setter
    def shared_parameters(self, values):
        with self._filter_parameter_context(shared=True):
            self.model.parameters = values

    @property
    def shared_linear_parameters(self):
        with self._filter_parameter_context(shared=True):
            return self.model.linear_parameters

    @shared_linear_parameters.setter
    def shared_linear_parameters(self, values):
        with self._filter_parameter_context(shared=True):
            self.model.linear_parameters = values

    def _concatenate_evaluation(self, funcname, xdata=None):
        """Evaluate model

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        if xdata is None:
            xdata = self.xdata
        ret = xdata * 0.0
        for idx, model in self._iter_models(xdata):
            func = getattr(model, funcname)
            ret[idx] = func(xdata=xdata[idx])
        return ret

    def evaluate_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        return self._concatenate_evaluation("evaluate_fullmodel", xdata=xdata)

    def evaluate_linear_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        return self._concatenate_evaluation("evaluate_linear_fullmodel", xdata=xdata)

    def evaluate_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        return self._concatenate_evaluation("evaluate_fitmodel", xdata=xdata)

    def evaluate_linear_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        return self._concatenate_evaluation("evaluate_linear_fitmodel", xdata=xdata)

    def derivative_fitmodel(self, param_idx, xdata=None):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        if xdata is None:
            xdata = self.xdata
        ret = xdata * 0.0
        idx_channels = self._idx_channels(len(xdata))
        for model_idx, param_idx in self._parameter_model_index(param_idx):
            idx = idx_channels[model_idx]
            model = self._models[model_idx]
            ret[idx] = model.derivative_fitmodel(param_idx, xdata=xdata[idx])
        return ret

    def linear_derivatives_fitmodel(self, xdata=None):
        """Derivates to all linear parameters

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
        """
        if xdata is None:
            xdata = self.xdata
        ret = numpy.empty((self.nlinear_parameters, xdata.size))
        for idx, model in self._iter_models(xdata):
            ret[:, idx] = model.linear_derivatives_fitmodel(xdata=xdata[idx])
        return ret

    def _iter_models(self, xdata):
        """Loop over the models and yield xdata slice

        :param array xdata:
        :yields (slice, Model):
        """
        for item in zip(self._idx_channels(len(xdata)), self._models):
            yield item

    def _idx_channels(self, nconcat):
        """Index of each model in the concatenated data

        :param int nconcat:
        :returns list(slice):
        """
        if self.caching_enabled:
            cache = self._cache.setdefault("idx_channels", {})
            if nconcat != cache.get("nconcat"):
                cache["idx"] = list(self._generate_idx_channels(nconcat))
                cache["nconcat"] = nconcat
            return cache["idx"]
        else:
            return list(self._generate_idx_channels(nconcat))

    def _generate_idx_channels(self, nconcat, stride=None):
        """Yield slice of the concatenated data for each model.
        The concatenated data could be sliced as `xdata[::stride]`.
        """
        nchannels = [m.nchannels for m in self._models]
        if not stride:
            stride, remain = divmod(sum(nchannels), nconcat)
            stride += remain > 0
        start = 0
        offset = 0
        i = 0
        for n in nchannels:
            # Index of model in concatenated xdata due to slicing
            stop = start + n
            lst = list(range(start + offset, stop, stride))
            nlst = len(lst)
            # Index of model in concatenated xdata after slicing
            idx = slice(i, i + nlst)
            i += nlst
            # Prepare for next model
            if lst:
                offset = lst[-1] + stride - stop
            else:
                offset -= n
            start = stop
            yield idx

    @contextmanager
    def caching_context(self):
        with ExitStack() as stack:
            ctx = super(ConcatModel, self).caching_context()
            stack.enter_context(ctx)
            for m in self._models:
                stack.enter_context(m.caching_context())
            yield
