from contextlib import contextmanager, ExitStack
import numpy
from PyMca5.PyMcaMath.linalg import lstsq
from PyMca5.PyMcaMath.fitting import Gefit

from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterModelBase
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterModel
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterModelManager


class LeastSquaresFitModelInterface:
    """All classes derived from `LeastSquaresFitModel` must implement
    this interface.
    """

    @property
    def xdata(self):
        raise NotImplementedError

    @xdata.setter
    def xdata(self, value):
        raise NotImplementedError

    @property
    def ydata(self):
        raise NotImplementedError

    @ydata.setter
    def ydata(self, value):
        raise NotImplementedError

    @property
    def ystd(self):
        return None

    @ystd.setter
    def ystd(self, value):
        raise NotImplementedError

    def evaluate_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        raise NotImplementedError

    def derivative_fitmodel(self, param_idx, xdata=None, **paramtype):
        """Derivate to a specific parameter of the fit model.

        Only required when you want to implement analytical derivatives
        of the fit model. Numerical derivatives are used by default.

        Note that the numerical derivatives for non-linear parameters
        are approximations. They are exact with arithmetic precission
        for linear parameters.

        :param int param_idx:
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        raise NotImplementedError

    def non_leastsquares_increment(self):
        """Only required when niter_non_leastsquares > 0"""
        raise NotImplementedError


class LeastSquaresFitModelBase(LeastSquaresFitModelInterface, ParameterModelBase):
    """A parameter model with least-squares optimization"""

    @property
    def ndata(self):
        raise NotImplementedError

    @property
    def yfitdata(self):
        raise NotImplementedError

    @property
    def yfitstd(self):
        raise NotImplementedError

    def evaluate_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        raise NotImplementedError

    def evaluate_linear_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: shape (ndata,)
        :returns array: n x ndata
        """
        raise NotImplementedError

    def evaluate_linear_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        derivatives = self.linear_derivatives_fitmodel(xdata=xdata)
        parameters = self.get_parameter_values(only_linear=True)
        return parameters.dot(derivatives)

    def linear_derivatives_fitmodel(self, xdata=None):
        """Derivates to all linear parameters

        :param array xdata: shape (ndata,)
        :returns array: shape (nparams, ndata)
        """
        nparams = self.get_n_parameters(only_linear=True)
        return numpy.array(
            [
                self.derivative_fitmodel(i, xdata=xdata, only_linear=True)
                for i in range(nparams)
            ]
        )

    def numerical_derivative_fitmodel(self, param_idx, xdata=None, **paramtype):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        raise NotImplementedError

    def compare_derivatives(self, xdata=None, **paramtype):
        """Compare analytical and numerical derivatives. Useful to
        validate the user defined `derivative_fitmodel`.

        :yields str, array, array: parameter name, analytical, numerical
        """
        for param_idx, name in enumerate(self.get_parameter_names(**paramtype)):
            ycalderiv = self.derivative_fitmodel(param_idx, xdata=xdata, **paramtype)
            ynumderiv = self.numerical_derivative_fitmodel(
                param_idx, xdata=xdata, **paramtype
            )
            yield name, ycalderiv, ynumderiv

    def linear_decomposition_fitmodel(self, xdata=None):
        """Linear decomposition of the fit model.

        :param array xdata: shape (ndata,)
        :returns array: nparams x ndata
        """
        derivatives = self.linear_derivatives_fitmodel(xdata=xdata)
        parameters = self.get_parameter_values(only_linear=True)
        return parameters[:, numpy.newaxis] * derivatives

    @property
    def yfullmodel(self):
        """Model of ydata"""
        return self.evaluate_fullmodel()

    @property
    def yfitmodel(self):
        """Model of yfitdata"""
        return self.evaluate_fitmodel()

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
        with self.__linear_fit_context():
            b = self.yfitdata
            for i in range(max(self.niter_non_leastsquares, 1)):
                A = self.linear_derivatives_fitmodel()
                result = lstsq(
                    A.T,  # ndata, nparams
                    b.copy(),  # ndata
                    uncertainties=True,
                    covariances=False,
                    digested_output=True,
                )
                if self.niter_non_leastsquares:
                    self.set_parameter_values(result["parameters"])
                    self.non_leastsquares_increment()
        result["linear"] = True
        result["parameters"] = result["parameters"]
        result["uncertainties"] = result["uncertainties"]
        result.pop("svd")
        return result

    def nonlinear_fit(self, full_output=False):
        """
        :param bool full_output: add statistics to fitted parameters
        :returns dict:
        """
        with self.__nonlinear_fit_context():
            constraints = self.get_parameter_constraints().T
            xdata = self.xdata
            ydata = self.yfitdata
            ystd = self.yfitstd
            for i in range(max(self.niter_non_leastsquares, 1)):
                parameters = self.get_parameter_values()
                result = Gefit.LeastSquaresFit(
                    self._gefit_evaluate_fitmodel,
                    parameters,
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
                    self.set_parameter_values(result[0])
                    self.non_leastsquares_increment()
        ret = {
            "linear": False,
            "parameters": result[0],
            "uncertainties": result[2],
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
    def __linear_fit_context(self):
        with ExitStack() as stack:
            ctx = self.linear_context(True)
            stack.enter_context(ctx)
            ctx = self._propertyCachingContext()
            stack.enter_context(ctx)
            ctx = self._linear_fit_context()
            stack.enter_context(ctx)
            yield

    @contextmanager
    def __nonlinear_fit_context(self):
        with ExitStack() as stack:
            ctx = self.linear_context(False)
            stack.enter_context(ctx)
            ctx = self._propertyCachingContext()
            stack.enter_context(ctx)
            ctx = self._nonlinear_fit_context()
            stack.enter_context(ctx)
            yield

    @contextmanager
    def _linear_fit_context(self):
        """To allow derived classes to add context"""
        yield

    @contextmanager
    def _nonlinear_fit_context(self):
        """To allow derived classes to add context"""
        yield

    def _gefit_evaluate_fitmodel(self, parameters, xdata):
        """Update parameters and evaluate model

        :param array parameters: shape (nparams,)
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        self.set_parameter_values(parameters)
        return self.evaluate_fitmodel(xdata=xdata)

    def _gefit_derivative_fitmodel(self, parameters, param_idx, xdata):
        """Update parameters and return derivate to a specific parameter

        :param array parameters: shape (nparams,)
        :param int param_idx:
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        self.set_parameter_values(parameters)
        return self.derivative_fitmodel(param_idx, xdata=xdata)

    def use_fit_result(self, result):
        """
        :param dict result:
        """
        self.set_parameter_values(result["parameters"], only_linear=result["linear"])

    @contextmanager
    def use_fit_result_context(self, result):
        """Changes the parameters only for the duration of this context

        :param dict result:
        """
        with self.linear_context(result["linear"]):
            with self._propertyCachingContext():
                self.use_fit_result(result)
                yield


class LeastSquaresFitModel(LeastSquaresFitModelBase, ParameterModel):
    """A least-squares parameter model which implement the fit model

    Derived classes:

        * implement the LeastSquaresFitModelInterface.
        * add parameter like a python property by using the `parameter_group` or
          `linear_parameter_group` decorators instead of the `property` decorator.

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

    @property
    def ndata(self):
        return len(self.xdata)

    @property
    def yfitdata(self):
        return self._y_full_to_fit(self.ydata)

    @property
    def yfitstd(self):
        return self._ystd_full_to_fit(self.ystd)

    def _y_full_to_fit(self, y, xdata=None):
        """Convert data from full model to fit model"""
        return y

    def _ystd_full_to_fit(self, ystd, xdata=None):
        """Convert standard deviation from full model to fit model"""
        return ystd

    def _y_fit_to_full(self, y, xdata=None):
        """Convert data from fit model to full model"""
        return y

    def evaluate_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        y = self.evaluate_fitmodel(xdata=xdata)
        return self._y_fit_to_full(y, xdata=xdata)

    def evaluate_linear_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        y = self.evaluate_linear_fitmodel(xdata=xdata)
        return self._y_fit_to_full(y, xdata=xdata)

    def derivative_fitmodel(self, param_idx, xdata=None, **paramtype):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        return self.numerical_derivative_fitmodel(param_idx, xdata=xdata, **paramtype)

    def numerical_derivative_fitmodel(self, param_idx, xdata=None, **paramtype):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        group = self._group_from_parameter_index(param_idx, **paramtype)
        param_is_linear = group.linear
        parameters = self.get_parameter_values(**paramtype)
        try:
            if param_is_linear:
                return self._numerical_derivative_linear_param(
                    parameters, param_idx, xdata=xdata, **paramtype
                )
            else:
                return self._numerical_derivative_nonlinear_param(
                    parameters, param_idx, xdata=xdata, **paramtype
                )
        finally:
            self.set_parameter_values(parameters, **paramtype)

    def _numerical_derivative_linear_param(
        self, parameters, param_idx, xdata=None, **paramtype
    ):
        """The numerical derivative to a linear parameter is exact
        within arithmetic precision.
        """
        # y(x) = p0*f0(x) + ... + pi*fi(x) + ...
        # dy/dpi(x) = fi(x)
        parameters = parameters.copy()
        for group in self._iter_parameter_groups(**paramtype):
            if group.linear:
                parameters[group.index] = 0
        parameters[param_idx] = 1
        self.set_parameter_values(parameters, **paramtype)
        return self.evaluate_fitmodel(xdata=xdata)

    def _numerical_derivative_nonlinear_param(
        self, parameters, param_idx, xdata=None, **paramtype
    ):
        """The numerical derivative to a non-linear parameter is an approximation"""
        # Choose delta to be a small fraction of the parameter value but not too small,
        # otherwise the derivative is zero.
        p0 = parameters[param_idx]
        delta = p0 * 1e-5
        if delta < 0:
            delta = min(delta, -1e-12)
        else:
            delta = max(delta, 1e-12)

        parameters = parameters.copy()
        parameters[param_idx] = p0 + delta
        self.set_parameter_values(parameters, **paramtype)
        f1 = self.evaluate_fitmodel(xdata=xdata)

        parameters[param_idx] = p0 - delta
        self.set_parameter_values(parameters, **paramtype)
        f2 = self.evaluate_fitmodel(xdata=xdata)

        return (f1 - f2) / (2.0 * delta)

    def compare_derivatives(self, xdata=None, **paramtype):
        """Compare analytical and numerical derivatives. Useful to
        validate the user defined `derivative_fitmodel`.

        :yields str, array, array: parameter name, analytical, numerical
        """
        for param_idx, name in enumerate(self.get_parameter_names(**paramtype)):
            ycalderiv = self.derivative_fitmodel(param_idx, xdata=xdata, **paramtype)
            ynumderiv = self.numerical_derivative_fitmodel(
                param_idx, xdata=xdata, **paramtype
            )
            yield name, ycalderiv, ynumderiv


class LeastSquaresCombinedFitModel(LeastSquaresFitModelBase, ParameterModelManager):
    """A least-squares parameter model which manages models that implement the fit model"""

    @property
    def ndata(self):
        return sum(model.ndata for model in self.models)

    @property
    def xdata(self):
        return self._get_concatenated_data("xdata")

    @xdata.setter
    def xdata(self, values):
        self._set_concatenated_data("xdata", values)

    @property
    def ydata(self):
        return self._get_concatenated_data("ydata")

    @ydata.setter
    def ydata(self, values):
        self._set_concatenated_data("ydata", values)

    @property
    def ystd(self):
        return self._get_concatenated_data("ystd")

    @ystd.setter
    def ystd(self, values):
        self._set_concatenated_data("ystd", values)

    @property
    def yfitdata(self):
        return self._get_concatenated_data("yfitdata")

    @property
    def yfitstd(self):
        return self._get_concatenated_data("yfitstd")

    def evaluate_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: shape (ndata,) or (nmodels, ndatai)
        :returns array: shape (ndata,) or (sum(ndatai),)
        """
        return self._concatenate_evaluation("evaluate_fullmodel", xdata=xdata)

    def evaluate_linear_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: shape (ndata,) or (nmodels, ndatai)
        :returns array: shape (ndata,) or (sum(ndatai),)
        """
        return self._concatenate_evaluation("evaluate_linear_fullmodel", xdata=xdata)

    def evaluate_fitmodel(self, xdata=None, _strided=False):
        """Evaluate the fit model.

        :param array xdata: shape (ndata,) or (nmodels, ndatai)
        :param bool _strided:
        :returns array: shape (ndata,) or (sum(ndatai),)
        """
        return self._concatenate_evaluation(
            "evaluate_fitmodel", xdata=xdata, strided=_strided
        )

    def derivative_fitmodel(self, param_idx, xdata=None, _strided=False, **paramtype):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: shape (ndata,)
        :param bool _strided:
        :returns array: shape (ndata,)
        """
        return self._get_concatenated_derivative(
            "derivative_fitmodel", param_idx, xdata=xdata, strided=_strided, **paramtype
        )

    def numerical_derivative_fitmodel(self, param_idx, xdata=None, **paramtype):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        return self._get_concatenated_derivative(
            "numerical_derivative_fitmodel", param_idx, xdata=xdata, **paramtype
        )

    def _gefit_evaluate_fitmodel(self, parameters, xdata):
        """Update parameters and evaluate model

        :param array parameters: shape (nparams,)
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        self.set_parameter_values(parameters)
        return self.evaluate_fitmodel(xdata=xdata, _strided=True)

    def _gefit_derivative_fitmodel(self, parameters, param_idx, xdata):
        """Update parameters and return derivate to a specific parameter

        :param array parameters: shape (nparams,)
        :param int param_idx:
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        self.set_parameter_values(parameters)
        return self.derivative_fitmodel(param_idx, xdata=xdata, _strided=True)

    def _get_concatenated_data(self, attr):
        """
        :param str attr:
        :returns array or None:
        """
        if self.nmodels == 0:
            return None
        data = [getattr(model, attr) for model in self.models]
        isnone = [d is None for d in data]
        if any(isnone):
            assert all(isnone)
            return None
        return numpy.concatenate(data)

    def _set_concatenated_data(self, attr, values):
        """
        :param str attr:
        :param array values:
        """
        if len(values) != self.ndata:
            raise ValueError("Not the expected number of channels")
        for model, values, _ in self._iter_model_data_slices(values):
            setattr(model, attr, values)

    def _concatenate_evaluation(self, funcname, xdata=None, strided=False):
        """Evaluate model

        :param array xdata: shape (ndata,) or (nmodels, ndatai)
        :param bool strided:
        :returns array: shape (ndata,) or (sum(ndatai),)
        """
        ret = numpy.empty(self._get_ndata(xdata, strided=strided))
        for model, xdata, idx in self._iter_model_data_slices(xdata, strided=strided):
            func = getattr(model, funcname)
            ret[idx] = func(xdata=xdata)
        return ret

    def _get_concatenated_derivative(
        self, funcname, param_idx, xdata=None, strided=False, **paramtype
    ):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: shape (ndata,)
        :param bool strided:
        :returns array: shape (ndata,)
        """
        n = self._get_ndata(xdata, strided=strided)
        ret = numpy.zeros(n)
        group = self._group_from_parameter_index(param_idx, **paramtype)
        model = self._linked_key_to_instance(group.instance_key)

        param_idx0 = param_idx
        cached = self._in_property_caching_context()
        if not cached:
            parameter_index_in_group = group.parameter_index_in_group(param_idx0)
        for modeli, xdata, idx in self._iter_model_data_slices(xdata, strided=strided):
            if not group.linked and modeli is not model:
                continue
            if cached:
                param_idx = param_idx0
            else:
                groupi = modeli._group_from_parameter_name(
                    group.property_name, **paramtype
                )
                param_idx = groupi.start_index + parameter_index_in_group
            func = getattr(modeli, funcname)

            tmpgroup = modeli._group_from_parameter_index(param_idx, **paramtype)
            if tmpgroup is None:
                breakpoint()
            ret[idx] = func(param_idx, xdata=xdata, **paramtype)

        return ret

    def _get_ndata(self, data, strided=False):
        """
        :param array data: shape (ndata,) or (nmodels, ndatai)
        :param bool strided:
        :returns int:
        """
        ndata = self.ndata
        if data is None:
            return ndata
        elif strided:
            return len(data)
        elif len(data) == ndata:
            return ndata
        else:
            if len(data) != self.nmodels:
                raise ValueError(f"Expected {self.nmodels} data arrays")
            return sum(len(x) for x in data)

    def _iter_model_data_slices(self, data, strided=False):
        """
        :param array data: shape (ndata,) or (nmodels, ndatai)
        :param bool strided:
        :yields tuple: model, datai, slice
        """
        i = 0
        if data is None:
            for model in self.models:
                n = model.ndata
                yield model, model.xdata, slice(i, i + n)
                i += n
        elif len(data) == self.ndata:
            for model in self.models:
                n = model.ndata
                idx = slice(i, i + n)
                yield model, data[idx], idx
                i += n
        elif strided:
            indices = self.__model_indices_after_slicing(data)
            for model, idx in zip(self.models, indices):
                yield model, data[idx], idx
        else:
            if len(data) != self.nmodels:
                raise ValueError(f"Expected {self.nmodels} data arrays")
            for model, xdata_model in zip(self.models, data):
                n = len(xdata_model)
                yield model, xdata_model, slice(i, i + n)
                i += n

    def __model_indices_after_slicing(self, data):
        ndata_models = [model.ndata for model in self.models]
        stride, remain = divmod(sum(ndata_models), len(data))
        stride += remain > 0
        start = 0
        offset = 0
        i = 0
        for n in ndata_models:
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
