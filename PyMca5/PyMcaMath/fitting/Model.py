import numpy
from PyMca5.PyMcaMath.fitting.ModelInterface import ModelInterface
from PyMca5.PyMcaMath.fitting.ModelParameterInterface import ModelParameterInterface


class Model(ModelInterface, ModelParameterInterface):
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
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        allp = cls._PARAMETER_GROUP_NAMES = list()
        linp = cls._LINEAR_PARAMETER_GROUP_NAMES = list()
        for name in sorted(dir(cls)):  # TODO: keep order if declaration?
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

    def _get_constraints(self, linear_only=None):
        """
        :param bool linear_only:
        :returns array: nparams x 3
        """
        if linear_only is None:
            linear_only = self.linear
        if linear_only:
            nparams = self.nlinear_parameters
        else:
            nparams = self.nparameters
        codes = numpy.zeros((nparams, 3))
        for group_name, idx in self._parameter_group_indices(linear_only=linear_only):
            paramprop = getattr(self.__class__, group_name)
            codes[idx] = paramprop.fconstraints(self)
        return codes

    def _get_parameters(self, linear_only=None):
        """
        :param bool linear_only:
        :returns array:
        """
        cache = self.getCache("parameters")
        if cache is None:
            return self._get_parameters_notcached(linear_only=linear_only)

        key = self._parameters_cache_key()
        parameters = cache.get(key, None)
        if parameters is None:
            parameters = cache[key] = self._get_parameters_notcached(
                linear_only=linear_only
            )
        return parameters

    def _set_parameters(self, params, linear_only=None):
        """
        :param bool linear_only:
        """
        cache = self.getCache("parameters")
        if cache is None:
            self._set_parameters_notcached(params, linear_only=linear_only)
        else:
            key = self._parameters_cache_key()
            cache[key] = params

    def _get_parameters_notcached(self, linear_only=None):
        """Helper for `_get_parameters`"""
        if linear_only is None:
            linear_only = self.linear
        if linear_only:
            nparams = self.nlinear_parameters
        else:
            nparams = self.nparameters
        params = numpy.zeros(nparams)
        for group_name, idx in self._parameter_group_indices(linear_only=linear_only):
            params[idx] = getattr(self, group_name)
        return params

    def _set_parameters_notcached(self, params, linear_only=None):
        """Helper of `_set_parameters`

        :param bool linear_only:
        """
        for group_name, idx in self._parameter_group_indices(linear_only=linear_only):
            setattr(self, group_name, params[idx])

    def _get_parameter(self, fget):
        """Helper for parameter getters."""
        parameters = self.getCache("parameters")
        if parameters is None:
            return fget(self)

        key = self._parameters_cache_key()
        parameters = parameters.get(key, None)
        if parameters is None:
            return fget(self)

        idx = self._parameter_group_index(fget.__name__)
        if idx is None:
            return fget(self)
        return parameters[idx]

    def _set_parameter(self, fset, value):
        """Helper for parameter setters"""
        parameters = self.getCache("parameters")
        if parameters is None:
            return fset(self, value)

        key = self._parameters_cache_key()
        parameters = parameters.get(key, None)
        if parameters is None:
            return fset(self, value)

        idx = self._parameter_group_index(fset.__name__)
        if idx is None:
            return fset(self, value)
        parameters[idx] = value

    def _parameter_groups(self, linear_only=None):
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

    def _parameter_groups_notcached(self, linear_only=None):
        """Helper for `_parameter_groups`.

        :param bool linear_only:
        :yields str, int: group name, nb. parameters in the group
        """
        if linear_only is None:
            linear_only = self.linear
        if linear_only:
            names = self.linear_parameter_group_names
        else:
            names = self.parameter_group_names
        for name in names:
            paramprop = getattr(self.__class__, name)
            n = paramprop.fcount(self)
            if n:
                yield name, n

    def _parameter_name_from_index(self, idx, linear_only=None):
        """Parameter index to group name and group index

        :returns str, int: group name, index in parameter group
        """
        i = 0
        for group_name, n in self._parameter_groups(linear_only=linear_only):
            if idx >= i and idx < (i + n):
                return group_name, idx - i
            i += n

    def _parameter_group_index(self, name, linear_only=None):
        """Parameter group name to index range

        :returns int or slice or None: index of parameter group in all parameters
        """
        for group_name, idx in self._parameter_group_indices(linear_only=linear_only):
            if name == group_name:
                return idx
        return None

    def _parameter_group_indices(self, linear_only=None):
        """Parameter indices for each group

        :yields int or slice: index of parameter group in all parameters
        """
        i = 0
        for group_name, n in self._parameter_groups(linear_only=linear_only):
            if n == 1:
                yield group_name, i
            else:
                yield group_name, slice(i, i + n)
            i += n

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
        return y

    def _ystd_full_to_fit(self, ystd, xdata=None):
        return ystd

    def _y_fit_to_full(self, y, xdata=None):
        return y

    def evaluate_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        y = self.evaluate_fitmodel(xdata=xdata)
        return self._y_fit_to_full(y, xdata=xdata)

    def evaluate_linear_fullmodel(self, xdata=None):
        """Evaluate the full model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        y = self.evaluate_linear_fitmodel(xdata=xdata)
        return self._y_fit_to_full(y, xdata=xdata)

    def evaluate_linear_fitmodel(self, xdata=None):
        """Evaluate the fit model.

        :param array xdata: length nxdata
        :returns array: n x nxdata
        """
        derivatives = self.linear_derivatives_fitmodel(xdata=xdata)
        return self.linear_parameters.dot(derivatives)

    def linear_derivatives_fitmodel(self, xdata=None):
        """Derivates to all linear parameters

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
        """
        with self._linear_context(True):
            return numpy.array(
                [
                    self.derivative_fitmodel(i, xdata=xdata)
                    for i in range(self.nlinear_parameters)
                ]
            )

    def derivative_fitmodel(self, param_idx, xdata=None):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        return self.numerical_derivative_fitmodel(param_idx, xdata=xdata)

    def numerical_derivative_fitmodel(self, param_idx, xdata=None):
        """Derivate to a specific parameter of the fit model.

        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        linear = self.linear
        if not linear:
            name, _ = self._parameter_name_from_index(param_idx)
            linear = name in self._LINEAR_PARAMETER_GROUP_NAMES

        keep = parameters = self.fit_parameters
        parameters = parameters.copy()
        try:
            if linear:
                return self._numerical_derivative_linear_param(parameters, param_idx, xdata=xdata)
            else:
                return self._numerical_derivative_nonlinear_param(parameters, param_idx, xdata=xdata)
        finally:
            self.fit_parameters = keep

    def _numerical_derivative_linear_param(self, parameters, param_idx, xdata=None):
        """The numerical derivative to a linear parameter is exact so
        far as the calculation of the fit model itself is exact.
        """
        # y(x) = p0*f0(x) + ... + pi*fi(x) + ...
        # dy/dpi(x) = fi(x)
        if self.linear:
            # All of them are linear parameters
            parameters = numpy.zeros_like(parameters)
        else:
            # Only some of them are linear parameters
            for name, idx in self._parameter_group_indices():
                if name in self._LINEAR_PARAMETER_GROUP_NAMES:
                    parameters[idx] = 0
        parameters[param_idx] = 1
        self.fit_parameters = parameters
        return self.evaluate_fitmodel(xdata=xdata)

    def _numerical_derivative_nonlinear_param(self, parameters, param_idx, xdata=None):
        """The numerical derivative to a non-linear parameter is an approximation
        """
        # Choose delta to be a small fraction of the
        # parameter value but not too small, otherwise
        # the derivative is zero.
        p0 = parameters[param_idx]
        delta = p0 * 1e-5
        if delta < 0:
            delta = min(delta, -1e-12)
        else:
            delta = max(delta, 1e-12)

        parameters[param_idx] = p0 + delta
        self.fit_parameters = parameters
        f1 = self.evaluate_fitmodel(xdata=xdata)

        parameters[param_idx] = p0 - delta
        self.fit_parameters = parameters
        f2 = self.evaluate_fitmodel(xdata=xdata)
        
        return (f1 - f2) / (2.0 * delta)

    def compare_derivatives(self, xdata=None):
        """Compare analytical and numerical derivatives. Useful to
        validate the user defined `derivative_fitmodel`.

        :yields str, array, array: parameter name, analytical, numerical 
        """
        for param_idx, name in enumerate(self.fit_parameter_names):
            ycalderiv = self.derivative_fitmodel(param_idx, xdata=xdata)
            ynumderiv = self.numerical_derivative_fitmodel(param_idx, xdata=xdata)
            yield name, ycalderiv, ynumderiv
