import numpy
from collections.abc import Sequence, MutableMapping
from PyMca5.PyMcaMath.fitting.LinkedModel import LinkedModelContainer
from PyMca5.PyMcaMath.fitting.ModelInterface import ModelInterface
from PyMca5.PyMcaMath.fitting.Model import Model


class ConcatModel(LinkedModelContainer, ModelInterface):
    """Concatenated model with shared parameters"""

    def __init__(self, models, shared_attributes=None):
        if not isinstance(models, Sequence):
            models = [models]
        for model in models:
            if not isinstance(model, Model):
                raise ValueError("'models' must be a list of type 'Model'")

        super().__init__(models)

        self.__fixed_shared_attributes = {
            "linear",
            "niter_non_leastsquares",
            "_included_parameters",
            "_excluded_parameters",
        }
        self.shared_attributes = shared_attributes

    def _iter_context_managers(self, context_name):
        ctxmgr = getattr(super(), context_name, None)
        if ctxmgr is not None:
            yield ctxmgr
        for model in self._models:
            yield getattr(model, context_name)

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
            for model in self._models:
                setattr(model, name, value)
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
                for model in self._all_other_models:
                    assert id(value) == id(getattr(model, name)), name
            else:
                for model in self._all_other_models:
                    assert value == getattr(model, name), name

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
            return sum(model.ndata for model in self._models)

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
            return numpy.concatenate([getattr(model, attr) for model in self._models])

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
            for model in self._models:
                yield model

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

    def _get_parameters(self, linear_only=None):
        """
        :param bool linear_only:
        :returns array:
        """
        return numpy.concatenate(
            [
                model._get_parameters(linear_only=linear_only)
                for model in self._iter_parameter_models()
            ]
        )

    def _set_parameters(self, values, linear_only=None):
        """
        :paramm array values:
        :param bool linear_only:
        """
        if linear_only is None:
            linear_only = self.linear
        i = 0
        for model in self._iter_parameter_models():
            if linear_only:
                n = model.nlinear_parameters
            else:
                n = model.nparameters
            if n:
                model._set_parameters(values[i : i + n], linear_only=linear_only)
                i += n
        self.share_attributes()  # TODO: find a better way to share parameters

    def _get_constraints(self, linear_only=None):
        """
        :param bool linear_only:
        :returns array: nparams x 3
        """
        return numpy.concatenate(
            [
                model._get_constraints(linear_only=linear_only)
                for model in self._iter_parameter_models()
            ]
        )

    def _parameter_groups(self, linear_only=None):
        """Yield name and count of enabled parameter groups

        :param bool linear_only:
        :yields str, int: group name, nb. parameters in the group
        """
        with self._filter_parameter_context(shared=True):
            for item in self.shared_model._parameter_groups(linear_only=linear_only):
                yield item
        with self._filter_parameter_context(shared=False):
            for i, model in enumerate(self._models):
                for name, n in self.shared_model._parameter_groups(
                    linear_only=linear_only
                ):
                    yield name + str(i), n

    def _parameter_model_index(self, idx, linear_only=None):
        """Convert parameter index of ConcatModel to a parameter indices
        of the underlying models (only one when parameter is not shared).

        :param bool linear_only:
        :param int idx:
        :yields (int, int): model index, parameter index in this model
        """
        cache = self._getCache("fit", "parameter_model_index")
        if cache is None:
            yield from self._iter_parameter_index(idx, linear_only=linear_only)
            return

        it = cache.get(idx)
        if it is None:
            it = cache[idx] = list(
                self._iter_parameter_index(idx, linear_only=linear_only)
            )
        yield from it

    def _iter_parameter_index(self, idx, linear_only=None):
        """Convert parameter index of ConcatModel to a parameter indices
        of the underlying models (only one when parameter is not shared).

        :param bool linear_only:
        :param int idx:
        :yields (int, int): model index, parameter index in this model
        """
        if linear_only is None:
            linear_only = self.linear
        if linear_only:
            nshared = self.nshared_linear_parameters
        else:
            nshared = self.nshared_parameters
        shared_attributes = self.shared_attributes
        if idx < nshared:
            for i, model in enumerate(self._models):
                iglobal = 0
                imodel = 0
                for name, n in model._parameter_groups(linear_only=linear_only):
                    if name in shared_attributes:
                        if idx >= iglobal and idx < (iglobal + n):
                            yield i, imodel + idx - iglobal
                        iglobal += n
                    imodel += n
        else:
            iglobal = nshared
            for i, model in enumerate(self._models):
                imodel = 0
                for name, n in model._parameter_groups(linear_only=linear_only):
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
        ret = numpy.empty(len(xdata))
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
        ret = numpy.empty(len(xdata))
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
        ret = numpy.empty((self.nlinear_parameters, len(xdata)))
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
        cache = self._getCache("fit", "model_data_slices")
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
        ndata = [model.ndata for model in self._models]
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
