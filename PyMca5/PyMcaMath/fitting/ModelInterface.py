import numpy
from PyMca5.PyMcaMath.linalg import lstsq
from PyMca5.PyMcaMath.fitting import Gefit

from PyMca5.PyMcaMath.fitting.ModelParameterInterface import ModelParameterInterface


class ModelUserInterface:
    """The part of the interface for all fit models that needs
    to be implemented by all classes derived from `Model`.
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

        The call is forwarded to `numerical_derivative_fitmodel`
        in `Model` so it is not strictly necessary to implement
        this method. Note that the numerical derivative to a
        non-linear parameter is an approximation.

        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        raise NotImplementedError

    def non_leastsquares_increment(self):
        raise NotImplementedError





class ModelInterface(ModelParameterInterface, ModelUserInterface):
    """Interface for all fit models (Model and ConcatModel derived classes)."""

    @property
    def parameter_group_names(self):
        raise AttributeError from NotImplementedError

    @property
    def linear_parameter_group_names(self):
        raise AttributeError from NotImplementedError

    def _parameter_groups(self, linear_only=None):
        """Yield name and count of enabled parameter groups

        :param bool linear_only:
        :yields str, int: group name, nb. parameters in the group
        """
        raise NotImplementedError

    def _get_parameters(self, linear_only=None):
        """
        :param bool linear_only:
        :returns array:
        """
        raise NotImplementedError

    def _get_constraints(self, linear_only=None):
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

    def linear_derivatives_fitmodel(self, xdata=None):
        """Derivates to all linear parameters

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
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
        return self._get_parameters(linear_only=False)

    @property
    def linear_parameters(self):
        return self._get_parameters(linear_only=True)

    @property
    def fit_parameters(self):
        """`parameters` when `linear=False` or `linear_parameters` when `linear=True`"""
        return self._get_parameters()

    @property
    def constraints(self):
        return self._get_constraints(linear_only=False)

    @property
    def linear_constraints(self):
        return self._get_constraints(linear_only=True)

    @property
    def fit_constraints(self):
        return self._get_constraints()

    @parameters.setter
    def parameters(self, values):
        return self._set_parameters(values, linear_only=False)

    @linear_parameters.setter
    def linear_parameters(self, values):
        return self._set_parameters(values, linear_only=True)

    @fit_parameters.setter
    def fit_parameters(self, values):
        return self._set_parameters(values)

    @property
    def nparameters(self):
        return sum(n for _, n in self._parameter_groups(linear_only=False))

    @property
    def nlinear_parameters(self):
        return sum(n for _, n in self._parameter_groups(linear_only=True))

    @property
    def nfit_parameters(self):
        return sum(n for _, n in self._parameter_groups())

    @property
    def parameter_names(self):
        return list(self._iter_parameter_names(linear_only=False))

    @property
    def linear_parameter_names(self):
        return list(self._iter_parameter_names(linear_only=True))

    @property
    def fit_parameter_names(self):
        return list(self._iter_parameter_names())

    def _iter_parameter_names(self, linear_only=None):
        for group_name, n in self._parameter_groups(linear_only=linear_only):
            if n > 1:
                for i in range(n):
                    yield group_name + str(i)
            else:
                yield group_name

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
                    self.linear_parameters = result["parameters"]
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
                    self.parameters,
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
                    self.parameters = result[0]
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
    def __linear_fit_context(self):
        with ExitStack() as stack:
            ctx = self._cachingContext("fit")
            stack.enter_context(ctx)
            ctx = self._linear_context(True)
            stack.enter_context(ctx)
            ctx = self._cachingContext("parameters")
            stack.enter_context(ctx)
            ctx = self._linear_fit_context()
            yield

    @contextmanager
    def __nonlinear_fit_context(self):
        with ExitStack() as stack:
            ctx = self._cachingContext("fit")
            stack.enter_context(ctx)
            ctx = self._linear_context(False)
            stack.enter_context(ctx)
            ctx = self._cachingContext("parameters")
            stack.enter_context(ctx)
            ctx = self._nonlinear_fit_context()
            yield

    @contextmanager
    def _linear_fit_context(self):
        """To allow derived classes to add context"""
        yield

    @contextmanager
    def _nonlinear_fit_context(self):
        """To allow derived classes to add context"""
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
        self.parameters = parameters
        return self.evaluate_fitmodel(xdata=xdata)

    def _gefit_derivative_fitmodel(self, parameters, param_idx, xdata):
        """Update parameters and return derivate to a specific parameter

        :param array parameters: length nparams
        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        self.parameters = parameters
        return self.derivative_fitmodel(param_idx, xdata=xdata)

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
            with self._cachingContext("parameters"):
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

    def linear_decomposition_fitmodel(self, xdata=None):
        """Linear decomposition of the fit model.

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
        """
        derivatives = self.linear_derivatives_fitmodel(xdata=xdata)
        return self.linear_parameters[:, numpy.newaxis] * derivatives
