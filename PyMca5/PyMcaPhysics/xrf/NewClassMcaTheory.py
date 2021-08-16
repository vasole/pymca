import os
import sys
import copy
import logging
import warnings
from contextlib import contextmanager
import numpy

from PyMca5 import PyMcaDataDir
from PyMca5.PyMcaIO import ConfigDict
from PyMca5.PyMcaMath.fitting import SpecfitFuns
from PyMca5.PyMcaMath.fitting import Gefit
from PyMca5.PyMcaMath.fitting.model import nonlinear_parameter_group
from PyMca5.PyMcaMath.fitting.model import independent_linear_parameter_group
from PyMca5.PyMcaMath.fitting.model import dependent_linear_parameter_group
from PyMca5.PyMcaMath.fitting.model import LeastSquaresFitModel
from PyMca5.PyMcaMath.fitting.model import LeastSquaresCombinedFitModel
from PyMca5.PyMcaMath.fitting.model.PolynomialModels import LinearPolynomialModel
from PyMca5.PyMcaMath.fitting.model.PolynomialModels import ExponentialPolynomialModel
from PyMca5.PyMcaMath.fitting.model.LinkedModel import linked_property
from PyMca5.PyMcaMath.fitting.model.ParameterModel import ParameterType
from PyMca5.PyMcaMath.fitting.model.ParameterModel import AllParameterTypes

from . import Elements
from . import ConcentrationsTool

FISX = ConcentrationsTool.FISX
if FISX:
    FisxHelper = ConcentrationsTool.FisxHelper


_logger = logging.getLogger(__name__)


def defaultConfigFilename():
    dirname = PyMcaDataDir.PYMCA_DATA_DIR
    filename = os.path.join(dirname, "McaTheory.cfg")
    if not os.path.exists(filename):
        # Frozen version deals differently with the path
        dirname = os.path.dirname(dirname)
        filename = os.path.join(dirname, "McaTheory.cfg")
        if not os.path.exists(filename):
            if dirname.lower().endswith(".zip"):
                dirname = os.path.dirname(dirname)
                filename = os.path.join(dirname, "McaTheory.cfg")
    if os.path.exists(filename):
        return filename
    else:
        print("Cannot find file McaTheory.cfg")
        raise IOError("File %s does not exist" % filename)


class McaTheoryConfigApi:
    """API on top of an MCA configuration"""

    def __init__(self, initdict=None, filelist=None, **kw):
        if initdict is None:
            initdict = defaultConfigFilename()
        if os.path.exists(initdict.split("::")[0]):
            self.config = ConfigDict.ConfigDict(filelist=initdict)
        else:
            raise IOError("File %s does not exist" % initdict)
        self._overwriteConfig(**kw)
        self.attflag = kw.get("attenuatorsflag", 1)
        self.configure()
        super().__init__()

    def _overwriteConfig(self, **kw):
        if "config" in kw:
            self.config.update(kw["config"])
        cfgfit = self.config["fit"]
        cfgfit["sumflag"] = kw.get("sumflag", cfgfit["sumflag"])
        cfgfit["escapeflag"] = kw.get("escapeflag", cfgfit["escapeflag"])
        cfgfit["continuum"] = kw.get("continuum", cfgfit["continuum"])
        cfgfit["stripflag"] = kw.get("stripflag", cfgfit["stripflag"])
        cfgfit["maxiter"] = kw.get("maxiter", cfgfit["maxiter"])
        cfgfit["hypermetflag"] = kw.get("hypermetflag", cfgfit["hypermetflag"])

    def _addMissingConfig(self):
        """Add missing information to the configuration"""
        cfgroot = self.config

        cfgroot["userattenuators"] = cfgroot.get("userattenuators", {})
        cfgroot["multilayer"] = cfgroot.get("multilayer", {})
        cfgroot["materials"] = cfgroot.get("materials", {})
        cfgroot["concentrations"] = cfgroot.get("concentrations", {})

        cfgpeakshape = cfgroot["peakshape"]
        cfgpeakshape["eta_factor"] = cfgpeakshape.get("eta_factor", 0.02)
        cfgpeakshape["fixedeta_factor"] = cfgpeakshape.get("fixedeta_factor", 0)
        cfgpeakshape["deltaeta_factor"] = cfgpeakshape.get(
            "deltaeta_factor", cfgpeakshape["eta_factor"]
        )

        cfgfit = cfgroot["fit"]
        cfgfit["fitfunction"] = cfgfit.get("fitfunction", None)
        if cfgfit["fitfunction"] is None:
            if cfgfit["hypermetflag"]:
                cfgfit["fitfunction"] = 0
            else:
                cfgfit["fitfunction"] = 1
        cfgfit["linearfitflag"] = cfgfit.get("linearfitflag", 0)
        cfgfit["fitweight"] = cfgfit.get("fitweight", 1)
        cfgfit["deltaonepeak"] = cfgfit.get("deltaonepeak", 0.010)

        cfgfit["energy"], idx = self._normalizeEnergyListParam(
            cfgfit.get("energy", None)
        )
        cfgfit["energyweight"], _ = self._normalizeEnergyListParam(
            cfgfit.get("energyweight"), idx=idx, default=1.0
        )
        cfgfit["energyflag"], _ = self._normalizeEnergyListParam(
            cfgfit.get("energyweight"), idx=idx, default=1
        )
        cfgfit["energyscatter"], _ = self._normalizeEnergyListParam(
            cfgfit.get("energyweight"), idx=idx, default=1
        )
        cfgfit["scatterflag"] = cfgfit.get("scatterflag", 0)

        cfgfit["stripalgorithm"] = cfgfit.get("stripalgorithm", 0)
        cfgfit["snipwidth"] = cfgfit.get("snipwidth", 30)
        cfgfit["linpolorder"] = cfgfit.get("linpolorder", 6)
        cfgfit["exppolorder"] = cfgfit.get("exppolorder", 6)
        cfgfit["stripconstant"] = cfgfit.get("stripconstant", 1.0)
        cfgfit["stripwidth"] = int(cfgfit.get("stripwidth", 1))
        cfgfit["stripfilterwidth"] = int(cfgfit.get("stripfilterwidth", 5))
        cfgfit["stripiterations"] = int(cfgfit.get("stripiterations", 20000))
        cfgfit["stripanchorsflag"] = int(cfgfit.get("stripanchorsflag", 0))
        cfgfit["stripanchorslist"] = cfgfit.get("stripanchorslist", [0, 0, 0, 0])

        cfgdetector = cfgroot["detector"]
        cfgdetector["detene"] = cfgdetector.get("detene", 1.7420)
        cfgdetector["ethreshold"] = cfgdetector.get("ethreshold", 0.020)
        cfgdetector["nthreshold"] = cfgdetector.get("nthreshold", 4)
        cfgdetector["ithreshold"] = cfgdetector.get("ithreshold", 1.0e-07)

    def _normalizeEnergyListParam(self, lst, idx=None, default=0):
        if not isinstance(lst, list):
            lst = [lst]
        if idx is None:
            idx = [
                i for i, v in enumerate(lst) if v and not isinstance(v, (str, bytes))
            ]
        n = len(lst)
        lst = [lst[i] if i < n else default for i in idx]
        return lst, idx

    def _sourceLines(self):
        """
        :yields tuple: (energy, weight, scatter)
        """
        cfg = self.config["fit"]
        scatterflag = cfg["scatterflag"]
        for energy, flag, weight, scatter in zip(
            cfg["energy"],
            cfg["energyflag"],
            cfg["energyweight"],
            cfg["energyscatter"],
        ):
            if energy and flag:
                yield energy, weight, scatter and scatterflag

    def _scatterLines(self):
        """Source lines that are included in the fit model

        :yields tuple: (energy, weight)
        """
        for energy, weight, scatter in self._sourceLines():
            if scatter and energy > self.SCATTER_ENERGY_THRESHOLD:
                yield energy, weight

    @property
    def _maxEnergy(self):
        """
        :returns float or None:
        """
        energies = [energy for energy, _, _ in self._sourceLines()]
        if energies:
            return max(energies)
        else:
            return None

    @property
    def _nSourceLines(self):
        """
        :returns int:
        """
        return len(list(self._sourceLines()))

    @property
    def _nRayleighLines(self):
        """
        :returns int:
        """
        return len(list(self._scatterLines()))

    def _attenuators(
        self,
        matrix=False,
        detector=False,
        detectorfilters=False,
        beamfilters=False,
        detectorfunnyfilters=False,
        yielddisabled=False,
    ):
        """Does not yield anything by default

        :yields list:
        """
        cfg = self.config["attenuators"]
        for name, alist in cfg.items():
            if not alist[0] and not yielddisabled:
                continue
            name = name.upper()
            if name == "MATRIX":
                # Sample description
                # enable, formula, density, thickness, anglein, angleout, usescatteringangle, scatteringangle
                if matrix:
                    if len(alist) == 6:
                        alist += [0, alist[4] + alist[5]]
                    yield alist[1:]
            elif name == "DETECTOR":
                # Detector description
                # enable, formula, density, thickness
                if detector:
                    yield alist[1:]
            else:
                # Filter
                # enable, formula, density, thickness, funny
                if len(alist) == 4:
                    alist.append(1.0)
                if name.startswith("BEAMFILTER"):
                    if beamfilters:
                        yield alist[1:]
                elif abs(alist[4] - 1.0) > 1.0e-10:
                    if detectorfunnyfilters:
                        yield alist[1:]
                else:
                    if detectorfilters:
                        yield alist[1:]

    @property
    def _matrix(self):
        """
        :returns list or None: [formula, density, thickness, anglein, angleout, usescatteringangle, scatteringangle]
        """
        if not self.attflag:
            return
        lst = list(self._attenuators(matrix=True))
        if lst:
            return lst[0]

    @property
    def _angleIn(self):
        """Angle between sample surface and primary beam"""
        lst = list(self._attenuators(matrix=True, yielddisabled=True))
        if lst:
            return lst[0][3]
        else:
            _logger.warning("Sample incident angle set to 45 deg.")
            return 45.0

    @property
    def _angleOut(self):
        """Angle between sample surface and outgoing beam (emission or scattering)"""
        lst = list(self._attenuators(matrix=True, yielddisabled=True))
        if lst:
            return lst[0][4]
        else:
            _logger.warning("Sample incident angle set to 45 deg.")
            return 45.0

    @property
    def _scatteringAngle(self):
        """Angle between primary beam and outgoing beam (emission or scattering)"""
        lst = list(self._attenuators(matrix=True, yielddisabled=True))
        if lst:
            if lst[0][5]:
                return lst[0][6]
            else:
                return self._angleIn + self._angleOut
        else:
            _logger.warning("Scattering angle set to 90 deg.")
            return 90.0

    @property
    def _detector(self):
        """
        :returns list or None: [formula, density, thickness]
        """
        if not self.attflag:
            return
        lst = list(self._attenuators(detector=True))
        if lst:
            return lst[0]

    def _multilayer(self):
        """Yields the sample layers

        :yields list: [formula, density, thickness]
        """
        if not self.attflag:
            return
        matrix = self._matrix
        if matrix[0].upper() == "MULTILAYER":
            cfg = self.config["multilayer"]
            layerkeys = list(cfg.keys())
            layerkeys.sort()
            for layer in layerkeys:
                alist = cfg[layer]
                if alist[0]:
                    yield alist[1:]
        else:
            yield matrix

    def _userAttenuators(self):
        """
        :yields dict: {"energy": [], "transmission": []}
        """
        if not self.attflag:
            return
        cfg = self.config["userattenuators"]
        for tableDict in cfg.values():
            if tableDict["use"]:
                yield tableDict

    def _emissionGroups(self):
        """
        :yields list: [Z, symb, linegroupname]
        """
        cfg = self.config["peaks"]
        for element, peaks in cfg.items():
            symb = element.capitalize()
            Z = Elements.getz(symb)
            if isinstance(peaks, list):
                for peak in peaks:
                    yield [Z, symb, peak]
            else:
                yield [Z, symb, peaks]

    def _configureElementsModule(self):
        """Configure the globals in the Elements module"""
        for material, info in self.config["materials"].items():
            Elements.Material[material] = copy.deepcopy(info)
        maxenergy = self._maxEnergy
        for element in self.config["peaks"]:
            symb = element.capitalize()
            if maxenergy != Elements.Element[symb]["buildparameters"]["energy"]:
                Elements.updateDict(energy=maxenergy)
                break

    def configure(self, newdict=None):
        if newdict:
            self.config.update(newdict)
        self._initializeConfig()
        return copy.deepcopy(self.config)

    def _initializeConfig(self):
        self._addMissingConfig()
        self._configureElementsModule()

    @property
    def _hypermet(self):
        if self.config["fit"]["fitfunction"] == 0:
            return self.config["fit"]["hypermetflag"]
        else:
            return 0

    @property
    def _hypermetGaussian(self):
        return self._hypermet & 1

    @property
    def _hypermetShortTail(self):
        return (self._hypermet >> 1) & 1

    @property
    def _hypermetLongTail(self):
        return (self._hypermet >> 2) & 2

    @property
    def _hypermetStep(self):
        return (self._hypermet >> 3) & 3

    def _anchorsIndices(self):
        cfg = self.config["fit"]
        if not cfg["stripanchorsflag"] or not cfg["stripanchorslist"]:
            return
        ravelled = self.xdata
        for channel in cfg["stripanchorslist"]:
            if channel <= ravelled[0]:
                continue
            index = numpy.nonzero(ravelled >= channel)[0]
            if len(index):
                index = min(index)
                if index > 0:
                    yield index


class McaTheoryLegacyApi:
    def setdata(self, *args, **kw):
        warnings.warn("McaTheory.setdata deprecated, please use setData", FutureWarning)
        return self.setData(*args, **kw)

    @property
    def sigmay(self):
        return self.ystd

    @property
    def sigmay0(self):
        return self.ystd0

    def startfit(self, *args, **kw):
        warnings.warn(
            "McaTheory.startfit deprecated, please use startFit", FutureWarning
        )
        return self.startFit(*args, **kw)

    def setConfiguration(self, ddict):
        """
        The current fit configuration dictionary is updated, but not replaced,
        by the input dictionary.
        It returns a copy of the final fit configuration.
        """
        return self.configure(ddict)

    def getConfiguration(self):
        """
        returns a copy of the current fit configuration parameters
        """
        return self.configure()

    def getStartingConfiguration(self):
        """
        returns a copy of the current fit configuration parameters
        """
        return self.configure()

    def enableOptimizedLinearFit(self):
        warnings.warn(
            "McaTheory.enableOptimizedLinearFit is deprecated (does nothing)",
            FutureWarning,
        )

    def disableOptimizedLinearFit(self):
        warnings.warn(
            "McaTheory.enableOptimizedLinearFit is deprecated (does nothing)",
            FutureWarning,
        )

    @property
    def codes(self):
        return self.get_parameter_constraints()

    def specfitestimate(self, x, y, z, xscaling=1.0, yscaling=1.0):
        warnings.warn(
            "McaTheory.specfitestimate is deprecated (does nothing)", FutureWarning
        )


class McaTheoryDataApi(McaTheoryConfigApi):
    """Add API for handling a single XRF spectrum (MCA data)"""

    def __init__(self, **kw):
        # Original XRF spectrum
        self._ydata0 = None
        self._xdata0 = None
        self._std0 = None
        self._xmin0 = None
        self._xmax0 = None
        self._expotime0 = None

        # XRF spectrum to fit
        self._ydata = None
        self._xdata = None
        self._std = None
        self._lastDataCacheParams = None

        super().__init__(**kw)

    @property
    def xdata(self):
        """Sorted and sliced view of xdata0"""
        self._refreshDataCache()
        return self._xdata

    @property
    def ydata(self):
        """Sorted and sliced view of ydata0"""
        self._refreshDataCache()
        return self._ydata

    @property
    def ystd(self):
        """Sorted and sliced view of ystd0"""
        self._refreshDataCache()
        return self._ystd

    @property
    def xdata0(self):
        return self._xdata0

    @property
    def ydata0(self):
        return self._ydata0

    @property
    def ystd0(self):
        return self._ystd0

    def setData(self, *var, **kw):
        """
        Method to update the data to be fitted.
        It accepts several combinations of input arguments, the simplest to
        take into account is:

        setData(x, y, sigmay=None, xmin=None, xmax=None)

        x corresponds to the spectrum channels
        y corresponds to the spectrum counts
        sigmay is the uncertainty associated to the counts. If not given,
               Poisson statistics will be assumed. If the fit configuration
               is set to no weight, it will not be used.
        xmin and xmax define the limits to be considered for performing the fit.
               If the fit configuration flag self.config['fit']['use_limit'] is
               set, they will be ignored. If xmin and xmax are not given, the
               whole given spectrum will be considered for fitting.
        time (seconds) is the factor associated to the flux, only used when using
               a strategy based on concentrations
        """
        if "y" in kw:
            ydata0 = kw["y"]
        elif len(var) > 1:
            ydata0 = var[1]
        elif len(var) == 1:
            ydata0 = var[0]
        else:
            ydata0 = None

        if ydata0 is None:
            return 1
        else:
            ydata0 = numpy.ravel(ydata0)

        if "x" in kw:
            xdata0 = kw["x"]
        elif len(var) > 1:
            xdata0 = var[0]
        else:
            xdata0 = None

        if xdata0 is None:
            xdata0 = numpy.arange(len(ydata0))
        else:
            xdata0 = numpy.ravel(xdata0)

        if "sigmay" in kw:
            ystd0 = kw["sigmay"]
        elif "stdy" in kw:
            ystd0 = kw["stdy"]
        elif len(var) > 2:
            ystd0 = var[2]
        else:
            ystd0 = None

        if ystd0 is None:
            # Poisson noise
            valid = ydata0 > 0
            if valid.any():
                ystd0 = numpy.sqrt(abs(ydata0))
                ystd0[~valid] = ystd0[valid].min()
            else:
                ystd0 = numpy.ones_like(ydata0)
        else:
            ystd0 = numpy.ravel(ystd0)

        timeFactor = kw.get("time", None)
        self._expotime0 = timeFactor
        if timeFactor is None:
            if self.config["concentrations"].get("useautotime", False):
                if not self.config["concentrations"]["usematrix"]:
                    msg = "Requested to use time from data but not present!!"
                    raise ValueError(msg)
        elif self.config["concentrations"].get("useautotime", False):
            self.config["concentrations"]["time"] = timeFactor

        self._xmin0 = kw.get("xmin", self.xmin)
        self._xmax0 = kw.get("xmax", self.xmax)
        return self._refreshDataCache(xdata0=xdata0, ydata0=ydata0, ystd0=ystd0)

    @property
    def xmin(self):
        """From config (if enabled) or the last `setData` call"""
        cfgfit = self.config["fit"]
        if cfgfit["use_limit"]:
            return cfgfit["xmin"]
        else:
            return self._xmin0

    @property
    def xmax(self):
        """From config (if enabled) or the last `setData` call"""
        cfgfit = self.config["fit"]
        if cfgfit["use_limit"]:
            return cfgfit["xmax"]
        else:
            return self._xmax0

    def getLastTime(self):
        return self._expotime0

    @property
    def _dataCacheParams(self):
        """Any change in these parameter will invalidate the cache"""
        return self.xmin, self.xmax

    def _refreshDataCache(self, xdata0=None, ydata0=None, ystd0=None):
        """Cache sorted and sliced view of the original XRF spectrum data"""
        params = self._dataCacheParams
        if xdata0 is None and ydata0 is None and ystd0 is None:
            if self._lastDataCacheParams == params:
                return  # the cached data is still valid

        # Original XRF spectrum
        if xdata0 is None:
            xdata0 = self.xdata0
        if xdata0 is None or not xdata0.size:
            return 1
        if ydata0 is None:
            ydata0 = self.ydata0
        if ydata0 is None or ydata0.size != xdata0.size:
            return 1
        if ystd0 is None:
            ystd0 = self.ystd0
        if ystd0 is None or ystd0.size != xdata0.size:
            return 1

        # XRF spectrum view
        selection = numpy.isfinite(ydata0)
        xmin = self.xmin
        if xmin is not None:
            selection &= xdata0 >= xmin
        xmax = self.xmax
        if xmax is not None:
            selection &= xdata0 <= xmax
        if not selection.any():
            return 1

        # Cache the original XRF spectrum and its view
        idx = numpy.argsort(xdata0)[selection]
        self._xdata = xdata0[idx]
        self._ydata = ydata0[idx]
        self._ystd = ystd0[idx]
        self._xdata0 = xdata0
        self._ydata0 = ydata0
        self._ystd0 = ystd0
        self._lastDataCacheParams = params


class McaTheoryBackground(McaTheoryDataApi):
    """Model for the background of an XRF spectrum"""

    CONTINUUM_LIST = [
        None,
        "Constant",
        "Linear",
        "Parabolic",
        "Linear Polynomial",
        "Exp. Polynomial",
    ]

    def __init__(self, **kw):
        # Numerical background
        self._numBkg = None
        self._lastNumBkgCacheParams = None

        # Analytical background
        self._continuum = None
        self._continuumModel = None
        self._lastContinuumCacheParams = None

        super().__init__(**kw)

    @property
    def hasNumBkg(self):
        return self._ynumbkg is not None

    def ynumbkg(self, xdata=None):
        """Get the numerical background (as opposed to the analytical background)"""
        ybkg = self._ynumbkg
        if ybkg is None:
            if xdata is None:
                return numpy.zeros(self.ndata)
            else:
                return numpy.zeros(len(xdata))
        if xdata is not None:
            # The numerical background is calculated on self.xdata
            # so we need to interpolate on xdata
            try:
                binterp = numpy.allclose(xdata, self.xdata)
            except ValueError:
                binterp = True
            if binterp:
                ybkg = numpy.interp(xdata, self.xdata, ybkg)
        return ybkg

    @property
    def hasContinuum(self):
        return self.continuumModel is not None

    def ycontinuum(self, xdata=None):
        """Get the analytical background (as opposed to the numerical background)"""
        model = self.continuumModel
        if model is None:
            if xdata is None:
                return numpy.zeros(self.ndata)
            else:
                return numpy.zeros(len(xdata))
        return model.evaluate_fullmodel(xdata=xdata)

    @property
    def hasBackground(self):
        return self.hasNumBkg or self.hasContinuum

    def ybackground(self, xdata=None):
        """Get the total background"""
        return self.ycontinuum(xdata=xdata) + self.ynumbkg(xdata=xdata)

    @property
    def _ynumbkg(self):
        self._refreshNumBkgCache()
        return self._numBkg

    @property
    def continuumModel(self):
        self._refreshContinuumCache()
        return self._continuumModel

    @property
    def _numBkgCacheParams(self):
        """Any change in these parameter will invalidate the cache"""
        cfg = self.config["fit"]
        params = [
            "stripflag",
            "stripalgorithm",
            "stripfilterwidth",
            "stripanchorsflag",
            "stripanchorslist",
        ]
        if cfg["stripalgorithm"] == 1:
            params += ["snipwidth"]
        else:
            params += ["stripwidth", "stripconstant", "stripiterations"]
        params = [cfg[p] for p in params]
        params.append(id(self._lastDataCacheParams))
        return params

    def _refreshNumBkgCache(self):
        """Cache numerical background"""
        bkgparams = self._numBkgCacheParams
        if self._lastNumBkgCacheParams == bkgparams:
            return  # the cached data is still valid
        elif self.ydata is None:
            self._numBkg = None
        elif self.config["fit"]["stripflag"]:
            signal = self._smooth(self.ydata)
            anchorslist = list(self._anchorsIndices())
            if self.config["fit"]["stripalgorithm"] == 1:
                self._numBkg = self._snip(signal, anchorslist)
            else:
                self._numBkg = self._strip(signal, anchorslist)
        else:
            self._numBkg = numpy.zeros_like(self.ydata)
        self._lastNumBkgCacheParams = bkgparams

    @property
    def continuum(self):
        return self.config["fit"]["continuum"]

    @continuum.setter
    def continuum(self, value):
        self.config["fit"]["continuum_name"] = self.CONTINUUM_LIST[value]
        self.config["fit"]["continuum"] = value

    @property
    def continuum_name(self):
        try:
            return self.config["fit"]["continuum_name"]
        except AttributeError:
            return self.CONTINUUM_LIST[self.continuum]

    @continuum_name.setter
    def continuum_name(self, name):
        self.config["fit"]["continuum"] = self.CONTINUUM_LIST.index(name)
        self.config["fit"]["continuum_name"] = name

    def _initializeConfig(self):
        super()._initializeConfig()
        self.continuum = self.continuum  # verify continuum name and index

    @property
    def _continuumCacheParams(self):
        cfgfit = self.config["fit"]
        continuum = self.continuum_name
        params = [id(self._lastDataCacheParams), continuum]
        if continuum == "Linear Polynomial":
            params.append(cfgfit["linpolorder"])
        elif continuum == "Exp. Polynomial":
            params.append(cfgfit["exppolorder"])
        return params

    def _refreshContinuumCache(self):
        contparams = self._continuumCacheParams
        if self._lastContinuumCacheParams == contparams:
            return  # the cached data is still valid

        # Instantiate the continuum model
        continuum = self.continuum_name
        if continuum is None:
            model = None
        elif continuum == "Constant":
            model = LinearPolynomialModel(degree=0, maxiter=10)
        elif continuum == "Linear":
            model = LinearPolynomialModel(degree=1, maxiter=10)
        elif continuum == "Parabolic":
            model = LinearPolynomialModel(degree=2, maxiter=10)
        elif continuum == "Linear Polynomial":
            model = LinearPolynomialModel(
                degree=self.config["fit"]["linpolorder"], maxiter=10
            )
        elif continuum == "Exp. Polynomial":
            model = ExponentialPolynomialModel(
                degree=self.config["fit"]["exppolorder"], maxiter=40
            )
        else:
            raise ValueError("Unknown continuum {}".format(continuum))
        self._continuumModel = model

        # Estimate the polynomial coefficients by fitting the numerical background
        if model is not None:
            model.xdata = self.xpol
            model.ydata = self.ynumbkg()
            result = model.fit()
            model.use_fit_result(result)

        self._lastContinuumCacheParams = contparams

    @property
    def xpol(self):
        return self._channelsToXpol(self.xdata)

    def _channelsToXpol(self, x):
        raise NotImplementedError

    @property
    def continuum_coefficients(self):
        model = self.continuumModel
        if model is None:
            return list()
        else:
            return model.get_parameter_values()

    @continuum_coefficients.setter
    def continuum_coefficients(self, values):
        model = self.continuumModel
        if model is not None:
            model.set_parameter_values(values)


class McaTheory(McaTheoryBackground, McaTheoryLegacyApi, LeastSquaresFitModel):
    """Model for MCA data"""

    BAND_GAP = 0.00385  # keV, silicon
    GAUSS_SIGMA_TO_FWHM = 2 * numpy.sqrt(2 * numpy.log(2))  # 2.3548
    FULL_ATTENUATION = 1.0e-300  # intensity assumed to be zero
    SCATTER_ENERGY_THRESHOLD = 0.2  # keV

    def __init__(self, **kw):
        # TODO: done for some initialization of SpecfitFuns?
        SpecfitFuns.fastagauss([1.0, 10.0, 1.0], numpy.arange(10.0))
        self.useFisxEscape(flag=False, apply=False)

        # XRF line groups
        self._nLineGroups = 0
        self._lineGroups = []
        self._lineGroupAreas = []
        self._fluoRates = []
        self._escapeLineGroups = []
        self._lastAreasCacheParams = None

        # Misc
        self._last_fit_result = None

        super().__init__(**kw)

    def useFisxEscape(self, flag=None, apply=True):
        """Make sure the model uses fisx to calculate the escape peaks
        when possible.
        """
        if flag and FISX:
            if ConcentrationsTool.FisxHelper.xcom is None:
                FisxHelper.xcom = xcom = FisxHelper.getElementsInstance()
            else:
                xcom = ConcentrationsTool.FisxHelper.xcom
            if hasattr(xcom, "setEscapeCacheEnabled"):
                xcom.setEscapeCacheEnabled(1)
                self._useFisxEscape = True
            else:
                self._useFisxEscape = False
        else:
            self._useFisxEscape = False
        if apply:
            self.configure()

    def _initializeConfig(self):
        super()._initializeConfig()
        self.linearfitflag = self.config["fit"][
            "linearfitflag"
        ]  # synchronize parameter_types
        self._preCalculateParameterIndependent()
        self._preCalculateParameterDependent()

    def _preCalculateParameterDependent(self):
        """Pre-calculate things that depend on the fit parameters"""
        pass

    def _preCalculateParameterIndependent(self):
        """Pre-calculate things that do not depend on the fit parameters"""
        self._preCalculateLineGroups()

    def _preCalculateLineGroups(self):
        """Calculate fluorescence and escape rates for emission and scatter line groups"""
        self._fluoRates = self._calcFluoRates()
        self._calcFluoRateCorrections()

        # Line groups: nested lists
        #   line group
        #       -> emission/scattering line
        #          [energy, rate, line name]
        # This is a filtered and normalized form of `_fluoRates`
        self._lineGroups = list(self._getEmissionLines())
        self._lineGroups.extend(self._getScatterLines())
        self._lineGroupNames = list(self._getEmissionLineNames())
        self._lineGroupNames.extend(self._getScatterNames())
        self._nLineGroups = len(self._lineGroups)
        self._lineGroupAreas = numpy.ones(self._nLineGroups)

        # Escape line groups: nested lists
        #   line group
        #       -> emission/scattering line
        #           -> escape line
        #              [energy, rate, escape name]
        self._escapeLineGroups = [
            self._calcEscapePeaks([peak[0] for peak in peaks])
            for peaks in self._lineGroups
        ]

    def _peakProfileParams(
        self, hypermet=None, selected_groups=None, normalized_fit_parameters=False
    ):
        """Raw peak profile parameters of emission/scatter/escape peaks.
        All parameters are defined in the energy domain.

        :param int hypermet:
        :param list or None selected_groups: all groups when `None`
                                             no groups when empty list (npeaks == 0)
        :param bool normalized_fit_parameters:
        :returns array: npeaks x npeakparams
        """
        lineGroups = self._lineGroups
        escapeLineGroups = self._escapeLineGroups
        linegroup_areas = self.linegroup_areas
        if selected_groups is not None:
            if not isinstance(selected_groups, (list, tuple)):
                selected_groups = [selected_groups]
            lineGroups = [lineGroups[i] for i in selected_groups]
            escapeLineGroups = [escapeLineGroups[i] for i in selected_groups]
            linegroup_areas = [linegroup_areas[i] for i in selected_groups]
        if hypermet is None:
            hypermet = self._hypermet
        if normalized_fit_parameters:
            linegroup_areas = numpy.ones(len(linegroup_areas))

        npeaks = sum(len(group) for group in lineGroups)
        npeaks += sum(len(escgroup) for group in escapeLineGroups for escgroup in group)
        if hypermet:
            # area, position, fwhm, ST AreaR, ST SlopeR, LT AreaR, LT SlopeR, STEP HeightR
            npeakparams = 8
        else:
            # area, position, fwhm, eta
            npeakparams = 4
        parameters = numpy.zeros((npeaks, npeakparams))
        if not parameters.size:
            return parameters

        # Peak positions and areas
        i = 0
        for group, escgroup, grouparea in zip(
            lineGroups, escapeLineGroups, linegroup_areas
        ):
            if not escgroup:
                escgroup = [[]] * len(group)
            for (energy, rate, _), esclines in zip(group, escgroup):
                peakarea = rate * grouparea
                parameters[i, 0] = peakarea
                parameters[i, 1] = energy
                i += 1
                for escen, escrate, _ in esclines:
                    parameters[i, 0] = peakarea * escrate
                    parameters[i, 1] = escen
                    i += 1

        # Area parameters from channel to energy domain
        parameters[:, 0] *= self.gain

        # FWHM in the energy domain
        parameters[:, 2] = self._peakFWHM(parameters[:, 1])

        # Other peak shape parameters
        if hypermet:
            if normalized_fit_parameters:
                shapeparams = [
                    1,
                    self.st_sloperatio,
                    1,
                    self.lt_sloperatio,
                    1,
                ]
            else:
                shapeparams = [
                    self.st_arearatio,
                    self.st_sloperatio,
                    self.lt_arearatio,
                    self.lt_sloperatio,
                    self.step_heightratio,
                ]
        else:
            shapeparams = [self.eta_factor]
        for i, param in enumerate(shapeparams, 3):
            parameters[:, i] = param

        return parameters

    @independent_linear_parameter_group
    def linegroup_areas(self):
        """line group areas in the channel domain"""
        self._refreshAreasCache()
        return self._lineGroupAreas

    @linegroup_areas.setter
    def linegroup_areas(self, values):
        self._lineGroupAreas[:] = values

    @linegroup_areas.counter
    def linegroup_areas(self):
        return self._nLineGroups

    @linegroup_areas.constraints
    def linegroup_areas(self):
        fixed = Gefit.CFIXED, 0, 0
        positive = Gefit.CPOSITIVE, 0, 0
        return [positive if area else fixed for area in self.linegroup_areas]

    @property
    def linegroup_names(self):
        return self._lineGroupNames

    def _refreshAreasCache(self, force=False):
        params = self._areasCacheParams
        if self._lastAreasCacheParams == params and not force:
            return  # the cached data is still valid
        self._estimateLineGroupAreas()
        self._lastAreasCacheParams = params

    @property
    def _areasCacheParams(self):
        """Any change in these parameter will invalidate the cache"""
        zero = int(self.zero * 1000)
        gain = int(self.gain * 1000)
        return self._dataCacheParams + (zero, gain, id(self._lineGroupAreas))

    def _estimateLineGroupAreas(self):
        """Estimated line group areas in the channel domain"""
        if self.hasBackground:
            ydata = self.ydata - self.ybackground()
        else:
            ydata = self.ydata
        xenergy = self.xenergy
        emin = xenergy.min()
        emax = xenergy.max()
        factor = numpy.sqrt(2 * numpy.pi) / self.GAUSS_SIGMA_TO_FWHM / self.gain
        zeroarea = max(ydata) * 1e-7

        lineGroups = self._lineGroups
        escapeLineGroups = self._escapeLineGroups
        linegroup_areas = self._lineGroupAreas
        for i, (group, escgroup) in enumerate(zip(lineGroups, escapeLineGroups)):
            selected_energy, _ = self.__select_highest_peak(group, escgroup, emin, emax)
            if selected_energy:
                chan = numpy.abs(xenergy - selected_energy).argmin()
                height = ydata[chan]  # omit dividing by the rate
                fwhm = self._peakFWHM(selected_energy)  # energy domain
                linegroup_areas[i] = height * fwhm * factor  # Gaussian
            else:
                linegroup_areas[i] = 0  # no peak within [emin, emax]

    def __select_highest_peak(self, group, escgroup, emin, emax):
        if not escgroup:
            escgroup = [[]] * len(group)
        selected_energy = 0
        selected_rate = 0
        for (peakenergy, rate, _), esclines in zip(group, escgroup):
            if peakenergy >= emin and peakenergy <= emax:
                if rate > selected_rate:
                    selected_energy = peakenergy
                    selected_rate = rate
            for peakenergy, escrate, _ in esclines:
                if peakenergy >= emin and peakenergy <= emax:
                    if rate * escrate > selected_rate:
                        selected_energy = peakenergy
                        selected_rate = rate * escrate
        return selected_energy, selected_rate

    def klm_markers(self, xenergy=None):
        if xenergy is None:
            xenergy = self.xenergy
        emin = xenergy.min()
        emax = xenergy.max()
        factor = numpy.sqrt(2 * numpy.pi) / self.GAUSS_SIGMA_TO_FWHM / self.gain

        lineGroups = self._lineGroups
        escapeLineGroups = self._escapeLineGroups
        linegroup_areas = self._lineGroupAreas
        linegroup_names = self._lineGroupNames

        for group, escgroup, label, area in zip(
            lineGroups, escapeLineGroups, linegroup_names, linegroup_areas
        ):
            if not escgroup:
                escgroup = [[]] * len(group)
            peakenergy, rate = self.__select_highest_peak(group, escgroup, emin, emax)
            if not peakenergy:
                continue  # no peak within [emin, emax]
            fwhm = self._peakFWHM(peakenergy)  # energy domain
            height = (rate * area) / (fwhm * factor)
            yield peakenergy, height, label
            for (peakenergy, rate, _), esclines in zip(group, escgroup):
                fwhm = self._peakFWHM(peakenergy)  # energy domain
                height = (rate * area) / (fwhm * factor)
                yield peakenergy, height, None
                for peakenergy, escrate, _ in esclines:
                    fwhm = self._peakFWHM(peakenergy)  # energy domain
                    height = (rate * escrate * area) / (fwhm * factor)
                    yield peakenergy, height, None

    def plot(self, title=None, markers=False):
        import matplotlib.pyplot as plt

        plt.plot(self.xenergy, self.yfitdata, label="data")
        plt.plot(self.xenergy, self.yfitmodel, label="fit")
        if markers:
            from matplotlib.pyplot import cm

            colors = iter(cm.rainbow(numpy.linspace(0, 1, self._nLineGroups)))
            for energy, height, label in self.klm_markers():
                if label:
                    color = next(colors)
                    plt.text(energy, height, label, color=color)
                plt.plot([energy, energy], [0, height], label=label, color=color)
        plt.legend()
        if title:
            plt.title(title)
        plt.show()

    def estimate(self):
        self._refreshAreasCache(force=True)

    def _evaluatePeakProfiles(
        self,
        xdata=None,
        hypermet=None,
        fast=True,
        selected_groups=None,
        normalized_fit_parameters=False,
    ):
        """Summed peak profiles of emission/scatter/escape peaks

        :param array xdata: 1D array
        :param int or None hypermet:
        :param bool fast: use lookup table for calculating exponentials
        :param bool selected_groups:
        :param bool normalized_fit_parameters:
        :returns array: same shape as x
        """
        parameters = self._peakProfileParams(
            hypermet=hypermet,
            selected_groups=selected_groups,
            normalized_fit_parameters=normalized_fit_parameters,
        )
        if parameters.size == 0:
            if xdata is None:
                return numpy.zeros(self.ndata)
            else:
                return numpy.zeros(len(xdata))
        if xdata is None:
            xdata = self.xdata
        energy = self._channelsToEnergy(xdata)
        if hypermet is None:
            hypermet = self._hypermet
        if hypermet:
            if fast:
                return SpecfitFuns.fastahypermet(parameters, energy, hypermet)
            else:
                return SpecfitFuns.ahypermet(parameters, energy, hypermet)
        else:
            return SpecfitFuns.apvoigt(parameters, energy)

    def _peakFWHM(self, energy):
        """Calculate the FWHM of a peak in the energy domain"""
        return numpy.sqrt(
            self.noise * self.noise
            + self.BAND_GAP
            * energy
            * self.fano
            * self.GAUSS_SIGMA_TO_FWHM
            * self.GAUSS_SIGMA_TO_FWHM
        )

    def _getEmissionLines(self):
        """Yields a list of emission lines for each group with total
        rate of 1 and sorted by energy.

        :yields list: [[energy, rate, "name"],
                       [energy, rate, "name"],
                        ...]
        """
        for group in sorted(self._emissionGroups()):
            yield self._getGroupEmissionLines(*group)

    def _getEmissionLineNames(self):
        for _, symb, line in sorted(self._emissionGroups()):
            yield symb + " " + line

    def _getScatterLines(self):
        """Yields a list for scattering lines for each source line.

        :yields list: [[energy, 1.0, "Scatter %03d"]]
        """
        scatteringAngle = numpy.radians(self._scatteringAngle)
        angleFactor = 1.0 - numpy.cos(scatteringAngle)
        for i, (en_elastic, _) in enumerate(self._scatterLines()):
            en_inelastic = en_elastic / (1.0 + (en_elastic / 511.0) * angleFactor)
            yield [[en_elastic, 1.0, "Scatter %03d" % i]]
            yield [[en_inelastic, 1.0, "Scatter %03d" % i]]

    def _getScatterNames(self):
        for i in range(self._nRayleighLines):
            yield "elastic" + str(i)
            yield "inelastic" + str(i)

    def _calcFluoRates(self):
        """Fluorescence rate for each emission line of each element.
        Rate means fluorescence intensity divided by primary intensity.

        :returns None or dict:
        """
        if self._matrix:
            if self._maxEnergy:
                multilayer = list(self._multilayer())
                if not multilayer:
                    text = "Your matrix is not properly defined.\n"
                    text += "If you used the graphical interface,\n"
                    text += "Please check the MATRIX tab"
                    raise ValueError(text)

                emissiongroups = sorted(self._emissionGroups())
                energylist, weightlist, scatterlist = zip(*self._sourceLines())
                detector = self._detector
                attenuatorlist = list(self._attenuators(detectorfilters=True))
                userattenuatorlist = list(self._userAttenuators())
                funnyfilters = list(self._attenuators(detectorfunnyfilters=True))
                filterlist = list(self._attenuators(beamfilters=True))
                alphain = self._angleIn
                alphaout = self._angleOut
                return Elements.getMultilayerFluorescence(
                    multilayer,
                    energylist,
                    layerList=None,
                    weightList=weightlist,
                    fulloutput=1,
                    attenuators=attenuatorlist,
                    alphain=alphain,
                    alphaout=alphaout,
                    elementsList=emissiongroups,
                    cascade=True,
                    detector=detector,
                    funnyfilters=funnyfilters,
                    beamfilters=filterlist,
                    forcepresent=1,
                    userattenuators=userattenuatorlist,
                )
            else:
                text = "Invalid energy for matrix configuration.\n"
                text += "Please check your BEAM parameters."
                raise ValueError(text)
        else:
            if self._nSourceLines > 1:
                raise ValueError("Multiple energies require a matrix definition")
            else:
                return None

    def _calcFluoRateCorrections(self):
        """Higher-order interaction corrections on the fluorescence rates.
        This will not be needed once fisx replaces the Elements module.
        """
        fisxcfg = self.config["fisx"] = {}
        if not FISX:
            return
        secondary = self.config["concentrations"].get("usemultilayersecondary", False)
        if secondary:
            corrections = FisxHelper.getFisxCorrectionFactorsFromFitConfiguration(
                self.config, elementsFromMatrix=False
            )
            fisxcfg["corrections"] = corrections
            fisxcfg["secondary"] = secondary

    def _getGroupEmissionLines(self, Z, symb, groupname):
        """Return a list of emission lines with total rate of 1 and
        sorted by energy.

        :param int Z: atomic number
        :param str symb: for example "Fe"
        :param str groupname: for example "K"
        :returns list: [[energy, rate, "name"],
                        [energy, rate, "name"],
                         ...]
        """
        if self._fluoRates is None:
            groups = Elements.Element[symb]
        else:
            groups = self._fluoRates[0][symb]

        peaks = []
        lines = groups.get(groupname + " xrays", dict())
        if not lines:
            return peaks
        for line in lines:
            lineinfo = groups[line]
            if lineinfo["rate"] > 0.0:
                peaks.append([lineinfo["energy"], lineinfo["rate"], line])

        if self._fluoRates is None:
            self._applyAttenuation(peaks, symb)

        totalrate = sum(peak[1] for peak in peaks)
        if not totalrate:
            text = "Intensity of %s %s is zero\n" % (symb, groupname)
            text += "Too high attenuation?"
            raise ZeroDivisionError(text)
        for peak in peaks:
            peak[1] /= totalrate

        ethreshold = self.config["fit"]["deltaonepeak"]
        return Elements._filterPeaks(
            peaks,
            ethreshold=ethreshold,
            ithreshold=0.0005,
            nthreshold=None,
            keeptotalrate=True,
        )

    def _applyAttenuation(self, peaks, symb):
        """Apply attenuation of primary and secondary beams.
        No high-order interactions are taken into account.
        Only 1 primary beam energy can be used.
        """
        self._applyMatrixAttenuation(peaks, symb)
        self._applyBeamFilterAttenuation(peaks, symb)
        self._applyDetectorFilterAttenuation(peaks, symb)
        self._applyFunnyFilterAttenuation(peaks, symb)
        self._applyDetectorAttenuation(peaks, symb)
        for peak in peaks:
            if peak[1] < self.FULL_ATTENUATION:
                peak[1] = 0

    def _iterLinearAttenuation(self, energies, **kw):
        """Linear attenuation coefficients of matrix, detector, filters, ...

        :param list energies:
        :param **kw: select the attenuator type to include
        """
        for attenuator in self._attenuators(**kw):
            formula, density, thickness, funnyfactor = attenuator
            rhod = density * thickness
            mu = Elements.getMaterialMassAttenuationCoefficients(formula, 1.0, energies)
            if len(energies) != 1 and len(mu["total"]) == 1:
                mu = mu["total"] * len(energies)
            else:
                mu = mu["total"]
            mulin = rhod * numpy.array(mu)
            yield mulin, funnyfactor

    def _applyBeamFilterAttenuation(self, peaks, symb):
        energies = Elements.Element[symb]["buildparameters"]["energy"]
        if not energies:
            raise ValueError("Invalid excitation energy")

        for mulin, _ in self._iterLinearAttenuation(energies, beamfilter=True):
            transmission = numpy.exp(-mulin)
            for peak, frac in zip(peaks, transmission):
                peak[1] *= frac

    def _applyDetectorFilterAttenuation(self, peaks, symb):
        energies = [peak[0] for peak in peaks]
        for mulin, _ in self._iterLinearAttenuation(energies, detectorfilter=True):
            transmission = numpy.exp(-mulin)
            for peak, frac in zip(peaks, transmission):
                peak[1] *= frac

    def _applyFunnyFilterAttenuation(self, peaks, symb):
        firstfunnyfactor = None
        energies = [peak[0] for peak in peaks]
        for mulin, funnyfactor in self._iterLinearAttenuation(
            energies, detectorfunnyfilter=True
        ):
            if (funnyfactor < 0.0) or (funnyfactor > 1.0):
                text = (
                    "Funny factor should be between 0.0 and 1.0., got %g" % funnyfactor
                )
                raise ValueError(text)
            transmission = numpy.exp(-mulin)
            if firstfunnyfactor is None:
                # only has to be multiplied once!!!
                firstfunnyfactor = funnyfactor
                transmission = funnyfactor * transmission + (1.0 - funnyfactor)
            else:
                if abs(firstfunnyfactor - funnyfactor) > 0.0001:
                    text = "All funny type attenuators must have same opening fraction"
                    raise ValueError(text)
            for peak, frac in zip(peaks, transmission):
                peak[1] *= frac

    def _applyDetectorAttenuation(self, peaks, symb):
        energies = [peak[0] for peak in peaks]
        for mulin, _ in self._iterLinearAttenuation(energies, symb, detector=True):
            attenuation = 1.0 - numpy.exp(-mulin)
            for peak, frac in zip(peaks, attenuation):
                peak[1] *= frac

    def _applyMatrixAttenuation(self, peaks, symb):
        matrix = self._matrix
        if not matrix:
            return
        maxenergy = Elements.Element[symb]["buildparameters"]["energy"]
        if not maxenergy:
            raise ValueError("Invalid excitation energy")
        formula, density, thickness = matrix[:3]
        alphaIn = self._angleIn
        alphaOut = self._angleOut

        energies = [x[0] for x in peaks] + [maxenergy]
        mu = Elements.getMaterialMassAttenuationCoefficients(formula, 1.0, energies)
        sinAlphaIn = numpy.sin(numpy.radians(alphaIn))
        sinAlphaOut = numpy.sin(numpy.radians(alphaOut))
        sinRatio = sinAlphaIn / sinAlphaOut
        muSource = mu["total"][-1]
        muFluo = numpy.array(mu["total"][:-1])

        transmission = 1.0 / (muSource + muFluo * sinRatio)
        rhod = density * thickness
        if rhod > 0.0 and abs(sinAlphaIn) > 0.0:
            expterm = -(muSource / sinAlphaIn + muFluo / sinAlphaOut) * rhod
            transmission *= 1.0 - numpy.exp(expterm)

        for peak, frac in zip(peaks, transmission):
            peak[1] *= frac

    def _applyUserAttenuators(self, peaks):
        for userattenuator in self.config["userattenuators"]:
            if self.config["userattenuators"][userattenuator]["use"]:
                transmission = Elements.getTableTransmission(
                    self.config["userattenuators"][userattenuator],
                    [x[0] for x in peaks],
                )
                for peak, frac in zip(peaks, transmission):
                    peak[1] *= frac

    def _calcEscapePeaks(self, energies):
        """For each energy a list of escape peaks with total rate of 1
        and sorted by energy.

        :param list energies:
        :returns list: [[[energy, rate, "name"],
                         [energy, rate, "name"],
                         ...]]
        """
        if not self.config["fit"]["escapeflag"]:
            return []
        if self._useFisxEscape:
            _logger.debug("Using fisx escape ratio's")
            return self._calcFisxEscapeRatios(energies)
        else:
            return self._calcPymcaEscapeRatios(energies)

    def _calcFisxEscapeRatios(self, energies):
        xcom = FisxHelper.xcom
        detele = self.config["detector"]["detele"]
        detector_composition = Elements.getMaterialMassFractions([detele], [1.0])
        ethreshold = self.config["detector"]["ethreshold"]
        ithreshold = self.config["detector"]["ithreshold"]
        nthreshold = self.config["detector"]["nthreshold"]
        xcom.updateEscapeCache(
            detector_composition,
            energies,
            energyThreshold=ethreshold,
            intensityThreshold=ithreshold,
            nThreshold=nthreshold,
        )

        escape_peaks = []
        for energy in energies:
            epeaks = xcom.getEscape(
                detector_composition,
                energy,
                energyThreshold=ethreshold,
                intensityThreshold=ithreshold,
                nThreshold=nthreshold,
            )
            epeaks = [
                [epeakinfo["energy"], epeakinfo["rate"], name[:-3].replace("_", " ")]
                for name, epeakinfo in epeaks.items()
            ]
            epeaks = Elements._filterPeaks(
                epeaks,
                ethreshold=ethreshold,
                ithreshold=ithreshold,
                nthreshold=nthreshold,
                absoluteithreshold=True,
                keeptotalrate=False,
            )
            escape_peaks.append(epeaks)
        return escape_peaks

    def _calcPymcaEscapeRatios(self, energies):
        escape_peaks = []
        detele = self.config["detector"]["detele"]
        for energy in energies:
            peaks = Elements.getEscape(
                [detele, 1.0, 1.0],
                energy,
                ethreshold=self.config["detector"]["ethreshold"],
                ithreshold=self.config["detector"]["ithreshold"],
                nthreshold=self.config["detector"]["nthreshold"],
            )
            escape_peaks.append(peaks)
        return escape_peaks

    @property
    def xenergy(self):
        return self._channelsToEnergy(self.xdata)

    def _channelsToEnergy(self, x):
        return self.zero + self.gain * x

    def _channelsToXpol(self, x):
        return self.zero + self.gain * (x - x.mean())

    @independent_linear_parameter_group
    def linpol_coefficients(self):
        if isinstance(self.continuumModel, LinearPolynomialModel):
            return self.continuum_coefficients
        else:
            return list()

    @linpol_coefficients.setter
    def linpol_coefficients(self, values):
        if isinstance(self.continuumModel, LinearPolynomialModel):
            self.continuum_coefficients = values

    @linpol_coefficients.counter
    def linpol_coefficients(self):
        continuum = self.continuum_name
        if continuum is None:
            return 0
        elif continuum == "Constant":
            return 1
        elif continuum == "Linear":
            return 2
        elif continuum == "Parabolic":
            return 3
        elif continuum == "Linear Polynomial":
            return self.config["fit"]["linpolorder"] + 1
        else:
            return 0

    @nonlinear_parameter_group
    def exppol_coefficients(self):
        if isinstance(self.continuumModel, ExponentialPolynomialModel):
            return self.continuum_coefficients
        else:
            return list()

    @exppol_coefficients.setter
    def exppol_coefficients(self, values):
        if isinstance(self.continuumModel, ExponentialPolynomialModel):
            self.continuum_coefficients = values

    @exppol_coefficients.counter
    def exppol_coefficients(self):
        if self.continuum_name == "Exp. Polynomial":
            return self.config["fit"]["exppolorder"] + 1
        else:
            return 0

    def _snip(self, signal, anchorslist):
        """Apply SNIP filtering to a signal"""
        _logger.debug("CALCULATING SNIP")
        n = len(signal)
        if len(anchorslist):
            anchorslist.sort()
        else:
            anchorslist = [0, n - 1]

        bkg = 0.0 * signal
        lastAnchor = 0
        cfg = self.config["fit"]
        width = cfg["snipwidth"]
        for anchor in anchorslist:
            if (anchor > lastAnchor) and (anchor < len(signal)):
                bkg[lastAnchor:anchor] = SpecfitFuns.snip1d(
                    signal[lastAnchor:anchor], width, 0
                )
                lastAnchor = anchor
        if lastAnchor < len(signal):
            bkg[lastAnchor:] = SpecfitFuns.snip1d(signal[lastAnchor:], width, 0)
        return bkg

    def _strip(self, signal, anchorslist):
        """Apply STRIP filtering to a signal"""
        cfg = self.config["fit"]
        niter = cfg["stripiterations"]
        if niter <= 0:
            return numpy.full_like(signal, signal.min())

        _logger.debug("CALCULATING STRIP")
        if (niter > 1000) and (cfg["stripwidth"] == 1):
            bkg = SpecfitFuns.subac(
                signal, cfg["stripconstant"], niter / 20, 4, anchorslist
            )
            bkg = SpecfitFuns.subac(
                bkg, cfg["stripconstant"], niter / 4, cfg["stripwidth"], anchorslist
            )
        else:
            bkg = SpecfitFuns.subac(
                signal, cfg["stripconstant"], niter, cfg["stripwidth"], anchorslist
            )
            if niter > 1000:
                # make sure to get something smooth
                bkg = SpecfitFuns.subac(bkg, cfg["stripconstant"], 500, 1, anchorslist)
            else:
                # make sure to get something smooth but with less than
                # 500 iterations
                bkg = SpecfitFuns.subac(
                    bkg,
                    cfg["stripconstant"],
                    int(cfg["stripwidth"] * 2),
                    1,
                    anchorslist,
                )
        return bkg

    def _smooth(self, y):
        """Smooth a signal"""
        try:
            y = y.astype(dtype=numpy.float64)
            w = self.config["fit"]["stripfilterwidth"]
            ysmooth = SpecfitFuns.SavitskyGolay(y, w)
        except Exception:
            print("Unsuccessful Savitsky-Golay smoothing: %s" % sys.exc_info())
            raise
        if ysmooth.size > 1:
            fltr = [0.25, 0.5, 0.25]
            ysmooth[1:-1] = numpy.convolve(ysmooth, fltr, mode=0)
            ysmooth[0] = 0.5 * (ysmooth[0] + ysmooth[1])
            ysmooth[-1] = 0.5 * (ysmooth[-1] + ysmooth[-2])
        return ysmooth

    @nonlinear_parameter_group
    def zero(self):
        return self.config["detector"]["zero"]

    @zero.setter
    def zero(self, value):
        self.config["detector"]["zero"] = value

    @zero.constraints
    def zero(self):
        if self.config["detector"]["fixedzero"]:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.zero
            delta = self.config["detector"]["deltazero"]
            return Gefit.CQUOTED, value - delta, value + delta

    @nonlinear_parameter_group
    def gain(self):
        return self.config["detector"]["gain"]

    @gain.setter
    def gain(self, value):
        self.config["detector"]["gain"] = value

    @gain.constraints
    def gain(self):
        if self.config["detector"]["fixedgain"]:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.gain
            delta = self.config["detector"]["deltagain"]
            return Gefit.CQUOTED, value - delta, value + delta

    @nonlinear_parameter_group
    def noise(self):
        return self.config["detector"]["noise"]

    @noise.setter
    def noise(self, value):
        self.config["detector"]["noise"] = value

    @noise.constraints
    def noise(self):
        if self.config["detector"]["fixednoise"]:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.noise
            delta = self.config["detector"]["deltanoise"]
            return Gefit.CQUOTED, value - delta, value + delta

    @nonlinear_parameter_group
    def fano(self):
        return self.config["detector"]["fano"]

    @fano.setter
    def fano(self, value):
        self.config["detector"]["fano"] = value

    @fano.constraints
    def fano(self):
        if self.config["detector"]["fixedfano"]:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.fano
            delta = self.config["detector"]["deltafano"]
            return Gefit.CQUOTED, value - delta, value + delta

    @dependent_linear_parameter_group
    def sum(self):
        if self.config["fit"]["sumflag"]:
            return self.config["detector"]["sum"]
        else:
            return 0.0

    @sum.setter
    def sum(self, value):
        self.config["detector"]["sum"] = value

    @sum.constraints
    def sum(self):
        if self.config["detector"]["fixedsum"] or not self.config["fit"]["sumflag"]:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.sum
            delta = self.config["detector"]["deltasum"]
            return Gefit.CQUOTED, value - delta, value + delta

    @nonlinear_parameter_group
    def eta_factor(self):
        return self.config["peakshape"]["eta_factor"]

    @eta_factor.setter
    def eta_factor(self, value):
        self.config["peakshape"]["eta_factor"] = value

    @eta_factor.counter
    def eta_factor(self):
        if self._hypermet:
            return 0
        else:
            return 1

    @eta_factor.constraints
    def eta_factor(self):
        if self.config["peakshape"]["fixedeta_factor"]:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.eta_factor
            delta = self.config["peakshape"]["deltaeta_factor"]
            return Gefit.CQUOTED, value - delta, value + delta

    @dependent_linear_parameter_group
    def step_heightratio(self):
        if self._hypermetStep:
            return self.config["peakshape"]["step_heightratio"]
        else:
            return 0

    @step_heightratio.setter
    def step_heightratio(self, value):
        self.config["peakshape"]["step_heightratio"] = value

    @step_heightratio.counter
    def step_heightratio(self):
        if self._hypermet:
            return 1
        else:
            return 0

    @step_heightratio.constraints
    def step_heightratio(self):
        if self.config["peakshape"]["fixedstep_heightratio"] or not self._hypermetStep:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.step_heightratio
            delta = self.config["peakshape"]["deltastep_heightratio"]
            return Gefit.CQUOTED, value - delta, value + delta

    @nonlinear_parameter_group
    def lt_sloperatio(self):
        return self.config["peakshape"]["lt_sloperatio"]

    @lt_sloperatio.setter
    def lt_sloperatio(self, value):
        self.config["peakshape"]["lt_sloperatio"] = value

    @lt_sloperatio.counter
    def lt_sloperatio(self):
        if self._hypermet:
            return 1
        else:
            return 0

    @lt_sloperatio.constraints
    def lt_sloperatio(self):
        if self.config["peakshape"]["fixedlt_sloperatio"] or not self._hypermetLongTail:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.lt_sloperatio
            delta = self.config["peakshape"]["deltalt_sloperatio"]
            return Gefit.CQUOTED, value - delta, value + delta

    @dependent_linear_parameter_group
    def lt_arearatio(self):
        if self._hypermetLongTail:
            return self.config["peakshape"]["lt_arearatio"]
        else:
            return 0

    @lt_arearatio.setter
    def lt_arearatio(self, value):
        self.config["peakshape"]["lt_arearatio"] = value

    @lt_arearatio.counter
    def lt_arearatio(self):
        if self._hypermet:
            return 1
        else:
            return 0

    @lt_arearatio.constraints
    def lt_arearatio(self):
        if self.config["peakshape"]["fixedlt_arearatio"] or not self._hypermetLongTail:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.lt_arearatio
            delta = self.config["peakshape"]["deltalt_arearatio"]
            return Gefit.CQUOTED, value - delta, value + delta

    @dependent_linear_parameter_group
    def st_sloperatio(self):
        return self.config["peakshape"]["st_sloperatio"]

    @st_sloperatio.setter
    def st_sloperatio(self, value):
        self.config["peakshape"]["st_sloperatio"] = value

    @st_sloperatio.counter
    def st_sloperatio(self):
        if self._hypermet:
            return 1
        else:
            return 0

    @st_sloperatio.constraints
    def st_sloperatio(self):
        if (
            self.config["peakshape"]["fixedst_sloperatio"]
            or not self._hypermetShortTail
        ):
            return Gefit.CFIXED, 0, 0
        else:
            value = self.st_sloperatio
            delta = self.config["peakshape"]["deltast_sloperatio"]
            return Gefit.CQUOTED, value - delta, value + delta

    @dependent_linear_parameter_group
    def st_arearatio(self):
        if self._hypermetShortTail:
            return self.config["peakshape"]["st_arearatio"]
        else:
            return 0

    @st_arearatio.setter
    def st_arearatio(self, value):
        self.config["peakshape"]["st_arearatio"] = value

    @st_arearatio.counter
    def st_arearatio(self):
        if self._hypermet:
            return 1
        else:
            return 0

    @st_arearatio.constraints
    def st_arearatio(self):
        if self.config["peakshape"]["fixedst_arearatio"] or not self._hypermetShortTail:
            return Gefit.CFIXED, 0, 0
        else:
            value = self.st_arearatio
            delta = self.config["peakshape"]["deltast_arearatio"]
            return Gefit.CQUOTED, value - delta, value + delta

    def _convert_parameter_names(self, names):
        linegroup_names = None
        for name in names:
            if "linegroup_areas" in name:
                if linegroup_names is None:
                    linegroup_names = self.linegroup_names
                idx = int(name.replace("linegroup_areas", ""))
                name = linegroup_names[idx]
                if "inelastic" in name:
                    i = int(name.replace("inelastic", ""))
                    name = "Scatter Compton%03d" % i
                elif "elastic" in name:
                    i = int(name.replace("elastic", ""))
                    name = "Scatter Peak%03d" % i
                yield name
            elif "linpol_coefficients" in name:
                if self.continuum < self.CONTINUUM_LIST.index("Linear Polynomial"):
                    idx = int(name.replace("linpol_coefficients", ""))
                    if idx == 0:
                        yield "Constant"
                    elif idx == 1:
                        yield "1st Order"
                    elif idx == 2:
                        yield "2nd Order"
                else:
                    yield name.replace("linpol_coefficients", "A")
            elif "exppol_coefficients" in name:
                yield name.replace("linpol_coefficients", "A")
            elif name == "st_arearatio":
                yield "ST AreaR"
            elif name == "st_sloperatio":
                yield "ST SlopeR"
            elif name == "lt_arearatio":
                yield "LT AreaR"
            elif name == "lt_sloperatio":
                yield "LT SlopeR"
            elif name == "step_heightratio":
                yield "STEP HeightR"
            elif name == "eta_factor":
                yield "Eta Factor"
            else:
                yield name.capitalize()

    def get_parameter_names(self, **paramtype):
        return tuple(
            self._convert_parameter_names(super().get_parameter_names(**paramtype))
        )

    def evaluate_fitmodel(self, xdata=None):
        return self.mcatheory(xdata=xdata)

    def mcatheory(
        self,
        xdata=None,
        hypermet=None,
        continuum=True,
        summing=True,
        selected_groups=None,
        normalized_fit_parameters=False,
    ):
        """Evaluate to MCA model (does not include the numerical background)

        y(x) = ycont(P(x)) + A1*F1(E(x)) + A2*F2(E(x)) + ...

            x: MCA channels (positive integers)

            ycont(x) = 0                              # no analytical background
                     = c0 + c1*x + c2*x^2 + ...       # linear polynomial
                     = c0 * exp[c1*x + c2*x^2 + ...]  # exponential polynomial

            E(x) = zero + gain*x
            P(x) = E(x - <x>)

            Fi(x): emission, scatter or escape peak

        Hypermet:

            A*F(x) =   A * Gnorm(x, u, s)
                     + st_arearatio*A * Tnorm(x, u, s, st_sloperatio)
                     + lt_arearatio*A * Tnorm(x, u, s, lt_sloperatio)
                     + step_heightratio*A * u/(sqrt(2*pi)*s) * Snorm(x, u, s)

            A: the area of the gaussian part, not the entire peak

            Gaussian is normalized in ]-inf, inf[
            Gnorm(x, u, s) = exp[-(x-u)^2/(2*s^2)] / (sqrt(2*pi)*s)

            Step is normalized in [0, inf[
            Snorm(x, u, s) = erfc[(x-u)/(sqrt(2)*s)] / (2 * u)

            Tail is normalized in ]-inf, inf[
            Tnorm(x, u, s, r) =   erfc[(x-u)/(sqrt(2)*s) + s/(sqrt(2)*r)]
                                * exp[(x-u)/r + (s/r)^2/2] / (2 * r)

        :param array xdata:
        :param int hypermet:
        :param bool continuum:
        :param bool summing:
        :param list selected_groups:
        :param bool normalized_fit_parameters:
        :returns array:
        """
        # Emission lines, scatter peaks and escape peaks
        y = self._evaluatePeakProfiles(
            xdata=xdata,
            hypermet=hypermet,
            selected_groups=selected_groups,
            normalized_fit_parameters=normalized_fit_parameters,
        )

        # Analytical background
        if continuum and self.continuumModel is not None:
            y += self.ycontinuum(xdata=xdata)

        # Pile-up
        if summing and self.sum:
            y += self.ypileup(ymodel=y, xdata=xdata)

        return y

    def ypileup(self, ymodel=None, xdata=None, normalized_fit_parameters=False):
        """The ymodel contains the peaks and the continuum.
        Pileup is
        """
        pileupfactor = self.sum
        if not pileupfactor:
            if xdata is None:
                return numpy.zeros(self.ndata)
            else:
                return numpy.zeros(len(xdata))
        if ymodel is None:
            ymodel = self.mcatheory(xdata=xdata, summing=False)
        if xdata is None:
            xmin = self.xmin
        else:
            xmin = min(xdata)
        if normalized_fit_parameters:
            pileupfactor = 1
        return pileupfactor * SpecfitFuns.pileup(
            ymodel, int(xmin), self.zero, self.gain, 0
        )

    def _y_full_to_fit(self, y, xdata=None):
        """The fitting is done after subtracting the numerical background"""
        if self.hasNumBkg:
            y = y - self.ynumbkg(xdata=xdata)
        if self.parameter_types == ParameterType.independent_linear and self.hasPileUp:
            # Note: this is not strictly correct but we have
            # no other choice until the pileup is implemented
            # properly (emission line combinations instead of convolution)
            y = y - self.ypileup(xdata=xdata)
        return y

    @property
    def hasPileUp(self):
        return bool(self.sum)

    def _y_fit_to_full(self, y, xdata=None):
        """The numerical background is not included in the fit model"""
        if self.hasNumBkg:
            return y + self.ynumbkg(xdata=xdata)
        else:
            return y

    def derivative_fitmodel(self, param_idx, xdata=None, **paramtype):
        """Derivate to a specific nonlinear_parameter_group

        :param int param_idx:
        :param array xdata: shape (ndata,)
        :returns array: shape (ndata,)
        """
        group = self._group_from_parameter_index(param_idx, **paramtype)
        name = group.property_name
        if name == "st_arearatio":
            return self.mcatheory(
                xdata=xdata, hypermet=2, continuum=False, normalized_fit_parameters=True
            )
        elif name == "lt_arearatio":
            return self.mcatheory(
                xdata=xdata, hypermet=4, continuum=False, normalized_fit_parameters=True
            )
        elif name == "step_heightratio":
            return self.mcatheory(
                xdata=xdata, hypermet=8, continuum=False, normalized_fit_parameters=True
            )
        elif name == "linegroup_areas":
            i = group.parameter_index_in_group(param_idx)
            return self.mcatheory(
                selected_groups=[i],
                normalized_fit_parameters=True,
                xdata=xdata,
                continuum=False,
            )
        elif name == "sum":
            return self.ypileup(xdata=xdata, normalized_fit_parameters=True)
        elif name == "linpol" and False:
            model = self.continuumModel
            keep = model.get_parameter_values(**paramtype)
            try:
                model.set_parameter_values(self.continuum_coefficients, **paramtype)
                i = group.parameter_index_in_group(param_idx)
                return model.derivative_fitmodel(i, xdata=xdata, **paramtype)
            finally:
                model.set_parameter_values(keep, **paramtype)
        else:
            return self.numerical_derivative_fitmodel(
                param_idx, xdata=xdata, **paramtype
            )

    @property
    def maxiter(self):
        return self.config["fit"]["maxiter"]

    @property
    def deltachi(self):
        return self.config["fit"]["deltachi"]

    @property
    def weightflag(self):
        return self.config["fit"]["fitweight"]

    @linked_property
    def linearfitflag(self):
        linearfitflag = int(self.parameter_types == ParameterType.independent_linear)
        self.config["fit"]["linearfitflag"] = linearfitflag
        return linearfitflag

    @linearfitflag.setter
    def linearfitflag(self, value):
        if value:
            self.parameter_types = ParameterType.independent_linear
        else:
            self.parameter_types = AllParameterTypes

    @contextmanager
    def linear_context(self, linear=None):
        keep = self.linearfitflag
        if linear is not None:
            self.linearfitflag = linear
        try:
            yield
        finally:
            self.linearfitflag = keep

    @property
    def parameter_types(self):
        return super(type(self), type(self)).parameter_types.fget(self)

    @parameter_types.setter
    def parameter_types(self, value):
        super(type(self), type(self)).parameter_types.fset(self, value)
        self.config["fit"]["linearfitflag"] = int(
            bool(self.parameter_types & ParameterType.independent_linear)
        )

    @contextmanager
    def _custom_iterative_optimization_context(self):
        with super()._custom_iterative_optimization_context():
            if abs(self.zero) < 1.0e-10:
                self.zero = 0.0
            yield

    def startFit(self, digest=False, linear=None):
        """Fit with legacy output"""
        with self.linear_context(linear=linear):
            result = self.fit(full_output=True)
            self._last_fit_result = result
            if digest:
                return self._legacyresult(result), self.digestresult()
            else:
                return self._legacyresult(result)

    @staticmethod
    def _legacyresult(result):
        return (
            result["parameters"],
            result["chi2_red"],
            result["uncertainties"],
            result["niter"],
            result["lastdeltachi"],
        )

    def digestresult(self):
        with self.use_fit_result_context(self._last_fit_result):
            result = {
                "xdata": self.xdata,
                "energy": self.xenergy,
                "ydata": self.ydata,
                "continuum": self.ybackground(),
                "yfit": self.yfullmodel,
                "ypileup": self.ypileup(),
                "parameters": self.get_parameter_names(),
                "fittedpar": self._last_fit_result["parameters"],
                "chisq": self._last_fit_result["chi2_red"],
                "sigmapar": self._last_fit_result["uncertainties"],
                "config": self.config.copy(),
            }
            return result

    def imagingDigestResult(self):
        with self.use_fit_result_context(self._last_fit_result):
            return dict()


class MultiMcaTheory(LeastSquaresCombinedFitModel):
    def __init__(self, ndetectors=1):
        models = {f"detector{i}": McaTheory() for i in range(ndetectors)}
        super().__init__(models)
        # self._enable_property_link("concentrations")
