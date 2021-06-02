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


class ModelUserInterface:
    """The user part of the interface for all fit models. Need to be
    implemented by all classes derived from `Model`.
    """

    @property
    def xdata(self):
        raise AttributeError from NotImplementedError

    @xdata.setter
    def xdata(self, value):
        raise AttributeError from NotImplementedError

    @property
    def ydata(self):
        raise AttributeError from NotImplementedError

    @ydata.setter
    def ydata(self, value):
        raise AttributeError from NotImplementedError

    @property
    def ystd(self):
        raise AttributeError from NotImplementedError

    @ystd.setter
    def ystd(self, value):
        raise AttributeError from NotImplementedError

    @property
    def linear(self):
        raise AttributeError from NotImplementedError

    @linear.setter
    def linear(self, value):
        raise AttributeError from NotImplementedError

    def evaluate_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        raise NotImplementedError

    def derivative_fitmodel(self, param_idx, xdata=None):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        raise NotImplementedError

    def linear_derivatives_fitmodel(self, xdata=None):
        """Derivates to all linear parameters

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
        """
        raise NotImplementedError

    def non_leastsquares_increment(self):
        raise NotImplementedError


class Cached:
    def __init__(self):
        self._cache = dict()

    @contextmanager
    def cachingContext(self, cachename):
        reset = not self.cachingEnabled(cachename)
        if reset:
            self._cache[cachename] = dict()
        try:
            yield
        finally:
            if reset:
                del self._cache[cachename]

    def cachingEnabled(self, cachename):
        return cachename in self._cache

    def getCache(self, cachename, *subnames):
        if cachename in self._cache:
            ret = self._cache[cachename]
            for cachename in subnames:
                try:
                    ret = ret[cachename]
                except KeyError:
                    ret[cachename] = dict()
                    ret = ret[cachename]
            return ret
        else:
            return None

    @staticmethod
    def enableCaching(cachename):
        def decorator(method):
            @functools.wraps(method)
            def cache_wrapper(self, *args, **kw):
                with self.cachingContext(cachename):
                    return method(self, *args, **kw)

            return cache_wrapper

        return decorator


class parameter(property):
    """Usage:

    .. highlight:: python
    .. code-block:: python

        class MyClass(Model):

            def __init__(self):
                self._myparam = 0.

            @parameter
            def myparam(self):
                return self._myparam

            @myparam.setter  # optional
            def myparam(self, value):
                self._myparam = value

            @myparam.counter  # optional
            def myparam(self):
                return 1

            @myparam.constraints  # optional
            def myparam(self):
                return 1, 0, 0
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=""):
        if fget is not None:
            fget = self._param_getter(fget)
        if fset is not None:
            fset = self._param_setter(fset)
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)
        self.fcount = self._fcount_default()
        self.fconstraints = self._fconstraints_default()

    def getter(self, fget):
        if fget is not None:
            fget = self._param_getter(fget)
        return super().getter(fget)

    def setter(self, fset):
        if fset is not None:
            fset = self._param_setter(fset)
        return super().setter(fset)

    def counter(self, fcount):
        self.fcount = fcount
        return self

    def constraints(self, fconstraints):
        self.fconstraints = fconstraints
        return self

    @classmethod
    def _param_getter(cls, fget):
        @functools.wraps(fget)
        def wrapper(self):
            return self._get_parameter(fget)

        return wrapper

    @classmethod
    def _param_setter(cls, fset):
        @functools.wraps(fset)
        def wrapper(self, value):
            return self._set_parameter(fset, value)

        return wrapper

    def _fcount_default(self):
        def fcount(wself):
            try:
                return len(self.fget(wself))
            except TypeError:
                return 1

        return fcount

    def _fconstraints_default(self):
        def fconstraints(wself):
            return numpy.zeros((self.fcount(wself), 3))

        return fconstraints


class linear_parameter(parameter):
    pass


class ModelInterface(Cached, ModelUserInterface):
    """Interface for all fit models (Model and ConcatModel derived classes)."""

    @property
    def parameter_group_names(self):
        raise AttributeError from NotImplementedError

    @property
    def linear_parameter_group_names(self):
        raise AttributeError from NotImplementedError

    def _parameter_groups(self, linear_only=False):
        """Yield name and count of enabled parameter groups

        :param bool linear_only:
        :yields str, int: group name, nb. parameters in the group
        """
        raise NotImplementedError

    def _get_parameters(self, linear_only=False, fitting=True):
        """
        :param bool linear_only:
        :param bool fitting:
        :returns array:
        """
        raise NotImplementedError

    def _get_constraints(self, linear_only=False):
        """
        :param bool linear_only:
        :returns array: nparams x 3
        """
        raise NotImplementedError

    def evaluate_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        raise NotImplementedError

    def evaluate_linear_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        raise NotImplementedError

    def evaluate_linear_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        raise NotImplementedError

    @property
    def ndata(self):
        raise AttributeError from NotImplementedError

    @property
    def yfitdata(self):
        raise AttributeError from NotImplementedError

    @property
    def yfitstd(self):
        raise AttributeError from NotImplementedError

    @property
    def yfullmodel(self):
        """Model of ydata"""
        return self.evaluate_fullmodel()

    @property
    def yfitmodel(self):
        """Model of yfitdata"""
        return self.evaluate_fitmodel()

    @property
    def parameters(self):
        return self._get_parameters(fitting=False)

    @property
    def linear_parameters(self):
        return self._get_parameters(linear_only=True, fitting=False)

    @property
    def fit_parameters(self):
        return self._get_parameters(fitting=True)

    @property
    def linear_fit_parameters(self):
        return self._get_parameters(linear_only=True, fitting=True)

    @property
    def constraints(self):
        return self._get_constraints()

    @property
    def linear_constraints(self):
        return self._get_constraints(linear_only=True)

    @parameters.setter
    def parameters(self, values):
        return self._set_parameters(values, fitting=False)

    @fit_parameters.setter
    def fit_parameters(self, values):
        return self._set_parameters(values, fitting=True)

    @linear_parameters.setter
    def linear_parameters(self, values):
        return self._set_parameters(values, linear_only=True, fitting=False)

    @linear_fit_parameters.setter
    def linear_fit_parameters(self, values):
        return self._set_parameters(values, linear_only=True, fitting=True)

    @property
    def nparameters(self):
        return sum(n for _, n in self._parameter_groups())

    @property
    def nlinear_parameters(self):
        return sum(n for _, n in self._parameter_groups(linear_only=True))

    @property
    def parameter_names(self):
        return list(self._iter_parameter_names())

    @property
    def linear_parameter_names(self):
        return list(self._iter_parameter_names(linear_only=True))

    def _iter_parameter_names(self, linear_only=False):
        for name, n in self._parameter_groups(linear_only=linear_only):
            if n > 1:
                for i in range(n):
                    yield name + str(i)
            else:
                yield name

    def fit(self, full_output=False):
        """
        :param bool full_output: add statistics to fitted parameters
        :returns dict:
        """
        if self.linear:
            return self.linear_fit(full_output=full_output)
        else:
            return self.nonlinear_fit(full_output=full_output)

    def linear_fit(self, full_output=False):
        """
        :param bool full_output: add statistics to fitted parameters
        :returns dict:
        """
        with self._linear_fit_context():
            b = self.yfitdata  # ndata
            for i in range(max(self.niter_non_leastsquares, 1)):
                A = self.linear_derivatives_fitmodel().T  # ndata, nparams
                result = lstsq(
                    A,
                    b.copy(),
                    uncertainties=True,
                    covariances=False,
                    digested_output=True,
                )
                if self.niter_non_leastsquares:
                    self.linear_fit_parameters = result["parameters"]
                    self.non_leastsquares_increment()
        result["linear"] = True
        result["parameters"] = self._fit_to_linear_parameters(result["parameters"])
        result["uncertainties"] = self._fit_to_linear_uncertainties(
            result["uncertainties"]
        )
        result.pop("svd")
        return result

    def nonlinear_fit(self, full_output=False):
        """
        :param bool full_output: add statistics to fitted parameters
        :returns dict:
        """
        with self._nonlinear_fit_context():
            constraints = self.constraints.T
            xdata = self.xdata
            ydata = self.yfitdata
            ystd = self.yfitstd
            for i in range(max(self.niter_non_leastsquares, 1)):
                result = Gefit.LeastSquaresFit(
                    self._gefit_evaluate_fitmodel,
                    self.fit_parameters,
                    model_deriv=self._gefit_derivative_fitmodel,
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
                    self.fit_parameters = result[0]
                    self.non_leastsquares_increment()
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

    @property
    def maxiter(self):
        return 100

    @property
    def deltachi(self):
        return None

    @property
    def weightflag(self):
        return 0

    @property
    def niter_non_leastsquares(self):
        return 0

    @contextmanager
    def _linear_fit_context(self):
        with self.cachingContext("fit"):
            with self._linear_context(True):
                with self._cache_parameters_context():
                    yield

    @contextmanager
    def _nonlinear_fit_context(self):
        with self.cachingContext("fit"):
            with self._linear_context(False):
                with self._cache_parameters_context():
                    yield

    @contextmanager
    def _cache_parameters_context(self):
        with self.cachingContext("parameters"):
            yield

    @contextmanager
    def _linear_context(self, linear):
        keep = self.linear
        self.linear = linear
        try:
            yield
        finally:
            self.linear = keep

    def _gefit_evaluate_fitmodel(self, parameters, xdata):
        """Update parameters and evaluate model

        :param array parameters: length nparams
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        self.fit_parameters = parameters
        return self.evaluate_fitmodel(xdata=xdata)

    def _gefit_derivative_fitmodel(self, parameters, param_idx, xdata):
        """Update parameters and return derivate to a specific parameter

        :param array parameters: length nparams
        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        self.fit_parameters = parameters
        return self.derivative_fitmodel(param_idx, xdata=xdata)

    def linear_decomposition_fitmodel(self, xdata=None):
        """Linear decomposition of the fit model.

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
        """
        derivatives = self.linear_derivatives_fitmodel(xdata=xdata)
        return self.linear_parameters[:, numpy.newaxis] * derivatives

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

    def use_fit_result(self, result):
        """
        :param dict result:
        """
        if result["linear"]:
            self.linear_parameters = result["parameters"]
        else:
            self.parameters = result["parameters"]

    @contextmanager
    def use_fit_result_context(self, result):
        with self._linear_context(result["linear"]):
            with self._cache_parameters_context():
                self.use_fit_result(result)
                yield

    def _parameters_to_fit(self, params):
        return params

    def _linear_parameters_to_fit(self, params):
        return params

    def _fit_to_parameters(self, params):
        return params

    def _fit_to_linear_parameters(self, params):
        return params

    def _fit_to_linear_uncertainties(self, uncertainties):
        return uncertainties

    def _fit_to_uncertainties(self, uncertainties):
        return uncertainties


class Model(ModelInterface):
    """Evaluation and derivatives of a model to be used in least-squares fitting.

    Derived classes:

        * implement the ModelUserInterface.
        * add parameter like a python property by using the `parameter` or
          `linear_parameter` decorators instead of the `property` decortor.

    There is a "fit model" and a "full model". The full model describes the data,
    the fit model describes the pre-processed data (for example smoothed,
    numerical back subtracted, ...). By default the full model and the fit model
    are identical.

    Fitting is done with linear least-squares optimization (not iterative)
    or non-linear least-squares optimization (iterative). An outer loop of
    non-least-squares optimization can be enabled (iterative).

    Example:

    .. code-block:: python

        model = MyModel()
        model.xdata = xdata
        model.ydata = ydata
        model.ystd = ydata**0.5

        plt.plot(model.xdata, model.ydata, label="data")
        plt.plot(model.xdata, model.ymodel, label="initial")

        result = model.fit()
        model.use_fit_result(result)

        plt.plot(model.xdata, model.ymodel, label="fit")
    """

    def __init__(self):
        self._included_parameters = None  # for ConcatModel
        self._excluded_parameters = None  # for ConcatModel
        super(Model, self).__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        allp = cls._PARAMETER_GROUP_NAMES = list()
        linp = cls._LINEAR_PARAMETER_GROUP_NAMES = list()
        for name in sorted(dir(cls)):
            attr = getattr(cls, name)
            if isinstance(attr, parameter):
                allp.append(name)
                if isinstance(attr, linear_parameter):
                    linp.append(name)

    @property
    def parameter_group_names(self):
        return list(self._filter_parameter_names(self._PARAMETER_GROUP_NAMES))

    @property
    def linear_parameter_group_names(self):
        return list(self._filter_parameter_names(self._LINEAR_PARAMETER_GROUP_NAMES))

    def _filter_parameter_names(self, names):
        included = self._included_parameters
        excluded = self._excluded_parameters
        for name in names:
            if included is not None and name not in included:
                continue
            if excluded is not None and name in excluded:
                continue
            yield name

    def _get_parameters(self, linear_only=False, fitting=True):
        """
        :param bool linear_only:
        :param bool fitting:
        :returns array:
        """
        cache = self.getCache("parameters")
        if cache is None:
            return self._get_parameters_notcached(
                linear_only=linear_only, fitting=fitting
            )

        key = self._parameters_cache_key()
        parameters = cache.get(key, None)
        if parameters is None:
            parameters = cache[key] = self._get_parameters_notcached(
                linear_only=linear_only, fitting=fitting
            )
        return parameters

    def _get_parameters_notcached(self, linear_only=False, fitting=True):
        """Helper for `_get_parameters`"""
        if linear_only:
            nparams = self.nlinear_parameters
        else:
            nparams = self.nparameters
        params = numpy.zeros(nparams)
        i = 0
        for name, n in self._parameter_groups(linear_only=linear_only):
            params[i : i + n] = getattr(self, name)
            i += n
        if fitting:
            if linear_only:
                return self._linear_parameters_to_fit(params)
            else:
                return self._parameters_to_fit(params)
        else:
            return params

    def _get_parameter(self, fget):
        """Helper for parameter getters."""
        parameters = self.getCache("parameters")
        if parameters is None:
            return fget(self)

        key = self._parameters_cache_key()
        parameters = parameters.get(key, None)
        if parameters is None:
            return fget(self)

        idx = self._parameter_slice_from_groupname(fget.__name__)
        return parameters[idx]

    def _get_constraints(self, linear_only=False):
        """
        :param bool linear_only:
        :returns array: nparams x 3
        """
        if linear_only:
            nparams = self.nlinear_parameters
        else:
            nparams = self.nparameters
        codes = numpy.zeros((nparams, 3), numpy.float64)
        i = 0
        for name, n in self._parameter_groups(linear_only=linear_only):
            codes[i : i + n] = getattr(self.__class__, name).fconstraints(self)
            i += n
        return codes

    def _set_parameters(self, params, linear_only=False, fitting=False):
        """
        :param bool linear_only:
        :param bool fitting:
        """
        cache = self.getCache("parameters")
        if cache is None:
            self._set_parameters_notcached(
                params, linear_only=linear_only, fitting=fitting
            )
        else:
            key = self._parameters_cache_key()
            cache[key] = params

    def _set_parameters_notcached(self, params, linear_only=False, fitting=False):
        """Helper of `_set_parameters`

        :param bool linear_only:
        :param bool fitting:
        """
        if fitting:
            if linear_only:
                params = self._fit_to_linear_parameters(params)
            else:
                params = self._fit_to_parameters(params)
        i = 0
        for name, n in self._parameter_groups(linear_only=linear_only):
            if n > 1:
                getattr(self, name)[:] = params[i : i + n]
            elif n == 1:
                setattr(self, name, params[i])
            i += n

    def _set_parameter(self, fset, value):
        """Helper for parameter setters"""
        parameters = self.getCache("parameters")
        if parameters is None:
            return fset(self, value)

        key = self._parameters_cache_key()
        parameters = parameters.get(key, None)
        if parameters is None:
            return fset(self, value)

        idx = self._parameter_slice_from_groupname(fset.__name__)
        parameters[idx] = value

    def _parameter_groups(self, linear_only=False):
        """Yield name and count of enabled parameter groups

        :param bool linear_only:
        :yields str, int: group name, nb. parameters in the group
        """
        cache = self.getCache("fit", "parameter_groups")
        if cache is None:
            yield from self._parameter_groups_notcached(linear_only=linear_only)
            return

        key = self._parameters_cache_key()
        it = cache.get(key)
        if it is None:
            it = cache[key] = list(
                self._parameter_groups_notcached(linear_only=linear_only)
            )
        yield from it

    def _parameters_cache_key(self):
        a = self._included_parameters
        b = self._excluded_parameters
        if a is not None:
            a = tuple(sorted(a))
        if b is not None:
            b = tuple(sorted(b))
        return a, b

    def _parameter_groups_notcached(self, linear_only=False):
        """Helper for `_parameter_groups`.

        :param bool linear_only:
        :yields str, int: group name, nb. parameters in the group
        """
        if linear_only:
            names = self.linear_parameter_group_names
        else:
            names = self.parameter_group_names
        for name in names:
            param = getattr(self.__class__, name)
            n = param.fcount(self)
            if n:
                yield name, n

    def _parameter_name_from_index(self, idx, linear_only=False):
        """Parameter index to group name and group index

        :returns str, int: group name, index in parameter group
        """
        i = 0
        for name, n in self._parameter_groups(linear_only=linear_only):
            if idx >= i and idx < (i + n):
                return name, idx - i
            i += n

    def _parameter_slice_from_groupname(self, name, linear_only=False):
        """Parameter group name to index range

        :returns int or slice: slice of parameter group in all parameters
        """
        i = 0
        for _name, n in self._parameter_groups(linear_only=linear_only):
            if name == _name:
                if n == 1:
                    return i
                else:
                    return slice(i, i + n)
            i += n

    @property
    def ndata(self):
        return len(self.xdata)

    @property
    def yfitdata(self):
        return self._ydata_to_fit(self.ydata)

    @property
    def yfitstd(self):
        return self._ystd_to_fit(self.ystd)

    def _ydata_to_fit(self, ydata, xdata=None):
        return ydata

    def _ystd_to_fit(self, ystd, xdata=None):
        return ystd

    def _fit_to_ydata(self, yfit, xdata=None):
        return yfit

    def evaluate_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        y = self.evaluate_fitmodel(xdata=xdata)
        return self._fit_to_ydata(y, xdata=xdata)

    def evaluate_linear_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        y = self.evaluate_linear_fitmodel(xdata=xdata)
        return self._fit_to_ydata(y, xdata=xdata)

    def evaluate_linear_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        derivatives = self.linear_derivatives_fitmodel(xdata=xdata)
        return self.linear_parameters.dot(derivatives)


class ConcatModel(ModelInterface):
    """Concatenated model with shared parameters"""

    def __init__(self, models, shared_attributes=None):
        if not isinstance(models, Sequence):
            models = [models]
        for model in models:
            if not isinstance(model, Model):
                raise ValueError("'models' must be a list of type 'Model'")
        if len(set(type(m) for m in models)) > 1:
            raise ValueError("Multiple model types are currently not supported")
        self._models = models
        self.__fixed_shared_attributes = {
            "linear",
            "niter_non_leastsquares",
            "_included_parameters",
            "_excluded_parameters",
        }
        self.shared_attributes = shared_attributes
        super().__init__()

    @contextmanager
    def cachingContext(self, cachename):
        """Enter the same caching context for all models"""
        with ExitStack() as stack:
            ctx = super().cachingContext(cachename)
            stack.enter_context(ctx)
            for m in self._models:
                stack.enter_context(m.cachingContext(cachename))
            yield

    @property
    def nmodels(self):
        return len(self._models)

    @property
    def shared_model(self):
        """Model used to get/set shared attributes"""
        return self._models[0]

    @property
    def _all_other_models(self):
        """All models except for `shared_model`"""
        return self._models[1:]

    def __getattr__(self, name):
        """Get shared attribute"""
        if self.nmodels and name in self.shared_attributes:
            return getattr(self.shared_model, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        """Set the attributes of all models when shared"""
        if (
            name != "_models"
            and self.nmodels
            and hasattr(self.shared_model, name)
            and name in self.shared_attributes
        ):
            for m in self._models:
                setattr(m, name, value)
        else:
            super().__setattr__(name, value)

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
            value = getattr(self.shared_model, name)
            if isinstance(value, (Sequence, MutableMapping, numpy.ndarray)):
                for m in self._all_other_models:
                    assert id(value) == id(getattr(m, name)), name
            else:
                for m in self._all_other_models:
                    assert value == getattr(m, name), name

    def share_attributes(self, shared_attributes=None):
        """Ensure attributes are shared

        :param Sequence(str) shared_attributes:
        """
        if self.nmodels <= 1:
            return
        if shared_attributes is None:
            shared_attributes = self._shared_attributes
        model = self.shared_model
        adict = {name: getattr(model, name) for name in shared_attributes}
        for model in self._all_other_models:
            for name, value in adict.items():
                try:
                    setattr(model, name, value)
                except AttributeError:
                    pass  # no setter

    @property
    def ndata(self):
        nmodels = self.nmodels
        if nmodels == 0:
            return 0
        else:
            return sum(m.ndata for m in self._models)

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
            return getattr(self.shared_model, attr)
        elif getattr(self.shared_model, attr) is None:
            return None
        else:
            return numpy.concatenate([getattr(m, attr) for m in self._models])

    def _set_data(self, attr, values):
        """
        :param str attr:
        :param array values:
        """
        if len(values) != self.ndata:
            raise ValueError("Not the expected number of channels")
        for idx, model in self._iter_model_data_slices(values):
            setattr(model, attr, values[idx])

    @contextmanager
    def _filter_parameter_context(self, shared=True):
        keepex = self._excluded_parameters  # shared between all models
        keepin = self._included_parameters  # shared between all models
        try:
            if shared:
                if keepin:
                    self._included_parameters = list(
                        set(keepin) - set(self.shared_attributes)
                    )
                else:
                    self._included_parameters = self.shared_attributes
            else:
                if keepex:
                    self._excluded_parameters.extend(self.shared_attributes)
                else:
                    self._excluded_parameters = self.shared_attributes
            yield
        finally:
            self._excluded_parameters = keepex
            self._included_parameters = keepin

    def _iter_parameter_models(self):
        """Yields models which are in such a state that they have
        either shared or non-shared parameters enabled.
        """
        with self._filter_parameter_context(shared=True):
            yield self.shared_model
        with self._filter_parameter_context(shared=False):
            for m in self._models:
                yield m

    def _iter_model_data_slices_types(self):
        modeltypes = set()
        for model in self._models:
            modeltype = type(model)
            if modeltype not in modeltypes:
                modeltypes.add(modeltype)
                yield model

    @property
    def nshared_parameters(self):
        with self._filter_parameter_context(shared=True):
            return self.shared_model.nparameters

    @property
    def nshared_linear_parameters(self):
        with self._filter_parameter_context(shared=True):
            return self.shared_model.nlinear_parameters

    def _get_parameters(self, linear_only=False, fitting=True):
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

    def _get_constraints(self, linear_only=False):
        """
        :param bool linear_only:
        :returns array: nparams x 3
        """
        return numpy.concatenate(
            [
                m._get_constraints(linear_only=linear_only)
                for m in self._iter_parameter_models()
            ]
        )

    def _parameter_groups(self, linear_only=False):
        """Yield name and count of enabled parameter groups

        :param bool linear_only:
        :yields str, int: group name, nb. parameters in the group
        """
        with self._filter_parameter_context(shared=True):
            for item in self.shared_model._parameter_groups(linear_only=linear_only):
                yield item
        with self._filter_parameter_context(shared=False):
            for i, m in enumerate(self._models):
                for name, n in self.shared_model._parameter_groups(
                    linear_only=linear_only
                ):
                    yield name + str(i), n

    def _parameter_model_index(self, idx, linear_only=False):
        """Convert parameter index of ConcatModel to a parameter indices
        of the underlying models (only one when parameter is not shared).

        :param bool linear_only:
        :param int idx:
        :yields (int, int): model index, parameter index in this model
        """
        cache = self.getCache("fit", "parameter_model_index")
        if cache is None:
            yield from self._iter_parameter_index(idx, linear_only=linear_only)
            return

        it = cache.get(idx)
        if it is None:
            it = cache[idx] = list(
                self._iter_parameter_index(idx, linear_only=linear_only)
            )
        yield from it

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
            return self.shared_model.parameters

    @shared_parameters.setter
    def shared_parameters(self, values):
        with self._filter_parameter_context(shared=True):
            self.shared_model.parameters = values

    @property
    def shared_linear_parameters(self):
        with self._filter_parameter_context(shared=True):
            return self.shared_model.linear_parameters

    @shared_linear_parameters.setter
    def shared_linear_parameters(self, values):
        with self._filter_parameter_context(shared=True):
            self.shared_model.linear_parameters = values

    def _concatenate_evaluation(self, funcname, xdata=None):
        """Evaluate model

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        if xdata is None:
            xdata = self.xdata
        ret = xdata * 0.0
        for idx, model in self._iter_model_data_slices(xdata):
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
        model_data_slices = self._model_data_slices(len(xdata))
        for model_idx, param_idx in self._parameter_model_index(param_idx):
            idx = model_data_slices[model_idx]
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
        for idx, model in self._iter_model_data_slices(xdata):
            ret[:, idx] = model.linear_derivatives_fitmodel(xdata=xdata[idx])
        return ret

    def _iter_model_data_slices(self, xdata):
        """
        :param array xdata:
        :yields (slice, Model):
        """
        for item in zip(self._model_data_slices(len(xdata)), self._models):
            yield item

    def _model_data_slices(self, nconcat):
        """Slice of each model in the concatenated data

        :param int nconcat:
        :returns list(slice):
        """
        cache = self.getCache("fit", "model_data_slices")
        if cache is None:
            return list(self._generate_model_data_slices(nconcat))
        else:
            if nconcat != cache.get("nconcat"):
                cache["idx"] = list(self._generate_model_data_slices(nconcat))
                cache["nconcat"] = nconcat
            return cache["idx"]

    def _generate_model_data_slices(self, nconcat, stride=None):
        """Yield slice of the concatenated data for each model.
        The concatenated data could be sliced as `xdata[::stride]`.
        """
        ndata = [m.ndata for m in self._models]
        if not stride:
            stride, remain = divmod(sum(ndata), nconcat)
            stride += remain > 0
        start = 0
        offset = 0
        i = 0
        for n in ndata:
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
