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

import os
import sys
import copy
import logging
import warnings
import numpy

from PyMca5 import PyMcaDataDir
from PyMca5.PyMcaIO import ConfigDict
from PyMca5.PyMcaMath.fitting import SpecfitFuns
from PyMca5.PyMcaMath.fitting.Model import Model, ConcatModel

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
    def __init__(self, initdict=None, filelist=None, **kw):
        if initdict is None:
            initdict = defaultConfigFilename()
        if os.path.exists(initdict.split("::")[0]):
            self.config = ConfigDict.ConfigDict(filelist=initdict)
        else:
            raise IOError("File %s does not exist" % initdict)
        self._overwriteConfig(**kw)
        self.attflag = kw.get("attenuatorsflag", 1)

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
        """Source lines that are included in the fir model

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
    def _hypermetShortTail(self):
        return (self.config["fit"]["hypermetflag"] >> 1) & 1

    @property
    def _hypermetLongTail(self):
        return (self.config["fit"]["hypermetflag"] >> 2) & 2

    @property
    def _hypermetStep(self):
        return (self.config["fit"]["hypermetflag"] >> 3) & 3

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


class McaTheory(McaTheoryConfigApi, McaTheoryLegacyApi, Model):
    """Model for MCA data"""

    BAND_GAP = 0.00385  # keV, silicon
    GAUSS_SIGMA_TO_FWHM = 2.3548
    MAX_ATTENUATION = 1.0e-300
    SCATTER_ENERGY_THRESHOLD = 0.2  # keV

    def __init__(self, **kw):
        super(McaTheory, self).__init__(**kw)
        # TODO: done for some initialization of SpecfitFuns?
        SpecfitFuns.fastagauss([1.0, 10.0, 1.0], numpy.arange(10.0))
        self.useFisxEscape(False)

        # Original XRF spectrum
        self._ydata0 = None
        self._xdata0 = None
        self._std0 = None
        self._xmin0 = None
        self._xmax0 = None

        # XRF spectrum to fit
        self._ydata = None
        self._xdata = None
        self._std = None
        self._lastDataCacheParams = None

        # XRF spectrum background
        self._numBkg = None
        self._lastNumBkgCacheParams = None

        self._lastTime = None

        self.strategyInstances = {}

        self.__toBeConfigured = False
        self.__configure()

    def useFisxEscape(self, flag=None):
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
        # same output as calling configure but with the calling program
        # knowing what is going on (no warning)
        if self.__toBeConfigured:
            return copy.deepcopy(self.__originalConfiguration)
        else:
            return self.configure()

    def configure(self, newdict=None):
        if newdict in [None, {}]:
            if self.__toBeConfigured:
                _logger.debug(
                    "WARNING: This configuration is the one of last fit.\n"
                    "It does not correspond to the one of next fit."
                )
        else:
            self.config.update(newdict)
            self.__toBeConfigured = False
            self.__configure()
        return copy.deepcopy(self.config)

    def __configure(self):
        self._initializeConfig()
        self._preCalculateParameterIndependent()
        self._preCalculateParameterDependent()

    def _preCalculateParameterIndependent(self):
        self._fluoRates = self._calcFluoRates()
        self._calcFluoRateCorrections()

        # Line groups: nested lists
        #   line group
        #       -> emission/scattering line
        #           -> energy, rate, line name
        self._lineGroups = list(self._getEmissionLines())
        self._lineGroups.extend(self._getScatterLines())

        # Escape line groups: nested lists
        #   line group
        #       -> emission/scattering line
        #           -> escape line
        #               -> energy, rate, escape name
        self._escapeLineGroups = [
            self._calcEscapePeaks([peak[0] for peak in peaks])
            for peaks in self._lineGroups
        ]

    def _getEmissionLines(self):
        """Yields a list of emission lines for each group with total
        rate of 1 and sorted by energy.

        :yields list: [[energy, rate, "name"],
                       [energy, rate, "name"],
                        ...]
        """
        for group in sorted(self._emissionGroups()):
            yield self._getGroupEmissionLines(*group)

    def _getScatterLines(self):
        """Yields a list for scattering lines for each source line.

        :yields list: [[energy, 1.0, "Scatter %03d"]]
        """
        scatteringAngle = self._scatteringAngle * numpy.pi / 180.0
        angleFactor = 1.0 - numpy.cos(scatteringAngle)
        for i, (en_elastic, _) in enumerate(self._scatterLines()):
            en_inelastic = en_elastic / (1.0 + (en_elastic / 511.0) * angleFactor)
            name = "Scatter %03d" % i
            yield [[en_elastic, 1.0, name]]
            yield [[en_inelastic, 1.0, name]]

    def _preCalculateParameterDependent(self):
        pass

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
            if peak[1] < self.MAX_ATTENUATION:
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
        sinAlphaIn = numpy.sin(alphaIn * numpy.pi / 180.0)
        sinAlphaOut = numpy.sin(alphaOut * numpy.pi / 180.0)
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
    def xdata(self):
        """Sorted and sliced view of xdata0"""
        self._refreshDataCache()
        return self._xdata

    @property
    def nchannels(self):
        return len(self.xdata)

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
    def ynumbkg(self):
        """Get the numerical background (as opposed to the analytical background)"""
        self._refreshNumBkgCache()
        return self._numBkg

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
        if self.__toBeConfigured:
            _logger.debug("setData RESTORE ORIGINAL CONFIGURATION")
            self.configure(self.__originalConfiguration)

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
        self._lastTime = timeFactor
        if timeFactor is None:
            cfgfit = self.config["fit"]
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
        return self._lastTime

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

        # XRF spectrum selection
        selection = numpy.isfinite(ydata0)
        xmin = self.xmin
        if xmin is not None:
            selection &= xdata0 >= xmin
        xmax = self.xmax
        if xmax is not None:
            selection &= xdata0 <= xmax
        if not selection.any():
            return 1

        # Cache original and reformed XRF spectrum
        idx = numpy.argsort(xdata0)[selection]
        self._xdata = xdata0[idx]
        self._ydata = ydata0[idx]
        self._ystd = ystd0[idx]
        self._xdata0 = xdata0
        self._ydata0 = ydata0
        self._ystd0 = ystd0
        self._lastDataCacheParams = params

        # Invalidate background cache
        self._lastNumBkgCacheParams = None

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

    def _snip(self, signal, anchorslist):
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
        cfg = self.config["fit"]
        niter = cfg["stripiterations"]
        if niter <= 0:
            return numpy.zeros_like(signal) + min(signal)

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

    @property
    def zero(self):
        return self.config["detector"]["zero"]

    @zero.setter
    def zero(self, value):
        self.config["detector"]["zero"] = value

    @property
    def gain(self):
        return self.config["detector"]["gain"]

    @gain.setter
    def gain(self, value):
        self.config["detector"]["gain"] = value

    @property
    def fano(self):
        return self.config["detector"]["fano"]

    @fano.setter
    def fano(self, value):
        self.config["detector"]["fano"] = value

    @property
    def sum(self):
        return self.config["detector"]["sum"]

    @sum.setter
    def sum(self, value):
        self.config["detector"]["sum"] = value

    @classmethod
    def _calc_fwhm(cls, noise, fano, energy):
        return numpy.sqrt(
            noise * noise
            + cls.BAND_GAP
            * energy
            * fano
            * cls.GAUSS_SIGMA_TO_FWHM
            * cls.GAUSS_SIGMA_TO_FWHM
        )

    @property
    def eta_factor(self):
        return self.config["peakshape"]["eta_factor"]

    @eta_factor.setter
    def eta_factor(self, value):
        self.config["peakshape"]["eta_factor"] = value

    @property
    def step_heightratio(self):
        return self.config["peakshape"]["step_heightratio"]

    @step_heightratio.setter
    def step_heightratio(self, value):
        self.config["peakshape"]["step_heightratio"] = value

    @property
    def lt_sloperatio(self):
        return self.config["peakshape"]["lt_sloperatio"]

    @lt_sloperatio.setter
    def lt_sloperatio(self, value):
        self.config["detector"]["lt_sloperatio"] = value

    @property
    def lt_arearatio(self):
        return self.config["peakshape"]["lt_arearatio"]

    @lt_arearatio.setter
    def lt_arearatio(self, value):
        self.config["peakshape"]["lt_arearatio"] = value

    @property
    def st_sloperatio(self):
        return self.config["peakshape"]["st_sloperatio"]

    @st_sloperatio.setter
    def st_sloperatio(self, value):
        self.config["peakshape"]["st_sloperatio"] = value

    @property
    def st_arearatio(self):
        return self.config["peakshape"]["st_arearatio"]

    @st_arearatio.setter
    def st_arearatio(self, value):
        self.config["peakshape"]["st_arearatio"] = value

    @property
    def _parameter_group_names(self):
        return [
            "zero",
            "gain",
            "noise",
            "fano",
            "sum",
            "st_arearatio",
            "st_sloperatio",
            "lt_arearatio",
            "lt_sloperatio",
            "step_heightratio",
            "eta_factor",
        ]

    @property
    def _linear_parameter_group_names(self):
        raise NotImplementedError

    def _iter_parameter_groups(self, linear_only=False):
        """
        :param bool linear_only:
        :yields (str, int): group name, nb. parameters in the group
        """
        if linear_only:
            names = self.linear_parameter_group_names
        else:
            names = self.parameter_group_names
        hypermet = self._hypermet
        for name in names:
            if name == "zero":
                yield name, 1
            elif name == "gain":
                yield name, 1
            elif name == "noise":
                yield name, 1
            elif name == "fano":
                yield name, 1
            elif name == "sum":
                yield name, 1
            elif name == "st_arearatio" and hypermet:
                yield name, 1
            elif name == "st_sloperatio" and hypermet:
                yield name, 1
            elif name == "lt_arearatio" and hypermet:
                yield name, 1
            elif name == "lt_sloperatio" and hypermet:
                yield name, 1
            elif name == "step_heightratio" and hypermet:
                yield name, 1
            elif name == "eta_factor" and not hypermet:
                yield name, 1
            else:
                raise ValueError(name)


class MultiMcaTheory(ConcatModel):
    def __init__(self, ndetectors=1):
        models = [McaTheory() for i in range(ndetectors)]
        shared_attributes = []
        super(MultiMcaTheory, self).__init__(
            models, shared_attributes=shared_attributes
        )
